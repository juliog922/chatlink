/* ChatLink Console — Vue 3 (CDN build, no tooling).
 *
 * Changes vs v2:
 * - The simulator is a CHAT: sendTurn() calls POST /api/test/turn and renders
 *   Kapa's real reply synchronously (full pipeline: gatekeeper → FSM → agent
 *   → RAG). No more "wait for the debounce" polling that always missed it.
 * - Gatekeeper drops and bot silence render as system lines in the feed.
 * - Conversation state (order status, cart, opt-out, latency) refreshes from
 *   the persisted session after every turn.
 * - fetchJson no longer swallows errors silently — failures surface as toasts.
 * - Background polling only where it still makes sense (health; feed refresh
 *   for channel/full modes where replies arrive from real transports).
 */
const { createApp, ref, onMounted, onUnmounted, computed, watch, nextTick } = Vue;

createApp({
    setup() {
        // ------------------------------------------------------------ auth
        // Soft gate for the ops console (same behaviour as before).
        const isAuthenticated = ref(false);
        const loginUsername = ref('');
        const loginPassword = ref('');
        const loginError = ref('');
        const SESSION_KEY = 'chatlink_session_expiry';
        const SESSION_MINUTES = 30;

        const performLogin = () => {
            if (loginUsername.value === 'admin' && loginPassword.value === 'kapalua') {
                isAuthenticated.value = true;
                loginError.value = '';
                localStorage.setItem(SESSION_KEY, Date.now() + SESSION_MINUTES * 60 * 1000);
                startBackgroundTasks();
            } else {
                loginError.value = 'Invalid username or password';
            }
        };

        // ---------------------------------------------------------- toasts
        const toasts = ref([]);
        let toastSeq = 0;
        const toast = (text, type = 'info') => {
            const id = ++toastSeq;
            toasts.value.push({ id, text, type });
            setTimeout(() => { toasts.value = toasts.value.filter(t => t.id !== id); }, 4000);
        };

        // ----------------------------------------------------------- state
        const currentView = ref('dashboard');
        const loading = ref(false);
        const health = ref({ status: '...', details: {} });
        const logs = ref([]);
        const logsFilter = ref({ level: '', search: '' });
        const connections = ref([]);

        // Simulator
        const simActors = ref([]);
        const simModes = ref({ modes: {}, linked_devices: [], full_mode_ready: false, full_mode_hint: '' });
        const sim = ref({
            channel: 'whatsapp',
            mode: 'logic',            // logic | channel (full = self-chat, nothing to inject)
            sender: '',
            receiver: '',             // empty -> backend routes to the Sim Salesman
            text: '',
            media_type: 'text',
            mock_client_force: true,  // fake the SAGE client if unknown (off = test the DROP path)
        });
        const simHistory = ref([]);   // persisted messages from /api/messages | /api/emails
        const simNotices = ref([]);   // ephemeral system lines (drops, silence) — not persisted
        const simState = ref({ order_status: '', confirmed_items: [], chat_context_summary: '', bot_enabled: true, last_benchmark_ms: 0, ctx: {} });

        // ------------------------------------------- simulation client cache
        // In-memory test clients (cache-only, checked BEFORE SAGE). CRUD here;
        // they also appear as selectable actors in the simulator.
        const simClients = ref([]);
        const newSimClient = ref({ identifier: '', name: '', code: '', notes: '', seed_orders: 0 });

        // Fake ORDER HISTORY for test clients. A test client resolves to
        // CodigoCliente TEST-xxxxxx, which does not exist in SAGE — so without
        // this its order history is empty and the history-resolution layer
        // cannot be exercised at all. Products are real catalog rows; only the
        // order statistics are invented. Memory only, cleared on restart.
        const simOrderSeed = ref({ identifier: '', n: 12, queries: '', seed: null, replace: true });
        const simOrders = ref({});          // identifier -> product rows
        const expandedSimClient = ref('');

        const loadSimClients = async () => {
            const d = await fetchJson('/api/test/clients');
            if (d) simClients.value = d.clients || [];
        };
        const addSimClient = async () => {
            const c = newSimClient.value;
            if (!c.identifier.trim()) { toast('Identifier (phone or email) is required', 'error'); return; }
            const d = await postJson('/api/test/clients', c);
            if (d && d.ok) {
                toast(`Test client saved: ${d.client.name}` +
                      (d.fake_orders ? ` · ${d.fake_orders} fake articles` : ''), 'success');
                newSimClient.value = { identifier: '', name: '', code: '', notes: '', seed_orders: 0 };
                loadSimClients();
            }
        };
        const deleteSimClient = async (identifier) => {
            const d = await fetchJson(`/api/test/clients/${encodeURIComponent(identifier)}`, { method: 'DELETE' });
            if (d && d.ok) { toast('Test client removed', 'success'); loadSimClients(); }
        };
        const clearSimClients = async () => {
            const d = await fetchJson('/api/test/clients', { method: 'DELETE' });
            if (d && d.ok) { toast(`Removed ${d.removed} test clients`, 'success'); loadSimClients(); }
        };

        const seedSimOrders = async () => {
            const s = simOrderSeed.value;
            if (!s.identifier) return toast('Pick a test client first', 'error');
            const body = {
                n: Number(s.n) || 12,
                queries: s.queries.split('\n').map(x => x.trim()).filter(Boolean),
                replace: !!s.replace,
                seed: s.seed === null || s.seed === '' ? null : Number(s.seed),
            };
            loading.value = true;
            try {
                const d = await postJson(
                    `/api/test/clients/${encodeURIComponent(s.identifier)}/orders`, body);
                if (d?.ok) {
                    toast(`Fake history: ${d.added} articles (${d.total} total)`, 'success');
                    simOrders.value = { ...simOrders.value, [s.identifier]: d.productos };
                    expandedSimClient.value = s.identifier;
                    loadSimClients();
                }
            } finally { loading.value = false; }
        };

        const loadSimOrders = async (identifier) => {
            if (expandedSimClient.value === identifier) { expandedSimClient.value = ''; return; }
            const d = await fetchJson(`/api/test/clients/${encodeURIComponent(identifier)}/orders`);
            if (d) {
                simOrders.value = { ...simOrders.value, [identifier]: d.productos || [] };
                expandedSimClient.value = identifier;
            }
        };

        const clearSimOrders = async (identifier) => {
            const d = await fetchJson(`/api/test/clients/${encodeURIComponent(identifier)}/orders`,
                                      { method: 'DELETE' });
            if (d) {
                toast('Fake history cleared', 'success');
                expandedSimClient.value = '';
                loadSimClients();
            }
        };

        /** Jump straight from a test client to the history diagnostics. */
        const inspectSimClient = (identifier) => {
            histIdentifier.value = identifier;
            currentView.value = 'clienthistory';
            nextTick(() => resolveIdentity());
        };

        // Self-chat: the salesman messaging their OWN number. The "client" is
        // that number, which is not in SAGE — so it lands on FAKE001 (or on
        // nothing at all in real full mode) and has no client code to hang a
        // history off. Registering it here makes every lookup resolve it
        // BEFORE SAGE, in simulation and in real full mode alike.
        const selfChat = ref({ identifier: '', n: 12, queries: '', seed: null,
                               replace: true, also_fake001: true });
        const selfChatState = ref(null);

        const loadSelfChat = async () => {
            const id = selfChat.value.identifier.trim();
            if (!id) { selfChatState.value = null; return; }
            const d = await fetchJson(`/api/test/self-chat/orders?identifier=${encodeURIComponent(id)}`);
            if (d) selfChatState.value = d;
        };

        const seedSelfChat = async () => {
            const s = selfChat.value;
            if (!s.identifier.trim()) return toast('Enter the number you self-chat with', 'error');
            loading.value = true;
            try {
                const d = await postJson('/api/test/self-chat/orders', {
                    identifier: s.identifier.trim(),
                    n: Number(s.n) || 12,
                    queries: s.queries.split('\n').map(x => x.trim()).filter(Boolean),
                    replace: !!s.replace,
                    also_fake001: !!s.also_fake001,
                    seed: s.seed === null || s.seed === '' ? null : Number(s.seed),
                });
                if (d?.ok) {
                    toast(`Self-chat ready: ${d.added} articles on ${d.codigos.join(' + ')}`, 'success');
                    await loadSelfChat();
                    loadSimClients();
                }
            } finally { loading.value = false; }
        };

        /** Erase the CURRENT simulator conversation: messages + session state.
         *  Distinct from "Fire timer now", which only skips the debounce wait.
         *  Works for real SAGE clients too, which the bulk reset cannot reach. */
        const clearSimConversation = async (alsoHistory = false) => {
            const receiver = sim.value.receiver || (sim.value.channel === 'whatsapp'
                ? '34600999001' : 'sales@sim.com');
            if (!sim.value.sender) return toast('Pick a sender first', 'error');
            if (!confirm(`Erase the conversation with ${sim.value.sender}?\n\n` +
                         'Deletes its stored messages and its session state (cart, unresolved ' +
                         'queue, memory)' + (alsoHistory ? ' and the cached order history' : '') +
                         '.\n\nSAGE is not touched.')) return;
            loading.value = true;
            try {
                const d = await postJson('/api/test/conversation/clear', {
                    channel: sim.value.channel, client: sim.value.sender, user: receiver,
                    borrar_historial_cache: !!alsoHistory,
                });
                if (d?.ok) {
                    toast(`Cleared: ${d.borrado.mensajes} messages, ${d.borrado.sesiones} session`, 'success');
                    simNotices.value = [];
                    loadSimChat();
                }
            } finally { loading.value = false; }
        };

        /** Wipe the traces of the test clients: chats, sessions, cached history.
         *  Scoped to simulation identifiers and TEST- / FAKE001 codes, so real
         *  traffic living in the same tables is never touched. */
        const simReset = ref({ identifier: '', borrar_clientes: false, incluir_fake001: true });
        const resetSimData = async () => {
            const r = simReset.value;
            const scope = r.identifier ? `for ${r.identifier}` : 'for ALL test clients';
            if (!confirm(`Reset simulation data ${scope}?\n\n` +
                         'Deletes their chat history, conversation sessions and cached order ' +
                         'history' + (r.incluir_fake001 ? ' (plus FAKE001)' : '') +
                         (r.borrar_clientes ? ', and removes the test client entries' : '') +
                         '.\n\nReal clients and SAGE are not touched.')) return;
            loading.value = true;
            try {
                const d = await postJson('/api/test/reset', {
                    identifiers: r.identifier ? [r.identifier] : [],
                    incluir_fake001: !!r.incluir_fake001,
                    borrar_clientes: !!r.borrar_clientes,
                });
                if (d?.ok) {
                    const b = d.borrado;
                    toast(`Reset: ${b.chats} chats, ${b.sesiones} sessions, ` +
                          `${b.historial_filas} history rows`, 'success');
                    selectedConv.value = null;
                    loadSimClients(); loadSelfChat(); loadDbOverview();
                }
            } finally { loading.value = false; }
        };

        const clearSelfChat = async () => {
            const id = selfChat.value.identifier.trim();
            if (!id) return;
            const d = await fetchJson(`/api/test/self-chat/orders?identifier=${encodeURIComponent(id)}`,
                                      { method: 'DELETE' });
            if (d?.ok) {
                toast(d.borrados.length ? `Cleared ${d.borrados.join(' + ')}` : 'Nothing to clear', 'success');
                await loadSelfChat();
                loadSimClients();
            }
        };

        // ------------------------------------------------------------ HTTP
        const fetchJson = async (url, opts) => {
            try {
                const res = await fetch(url, opts);
                if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
                return await res.json();
            } catch (e) {
                toast(`Request failed: ${url.split('?')[0]} (${e.message})`, 'error');
                return null;
            }
        };
        const postJson = (url, body) => fetchJson(url, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });

        // -------------------------------------------------------- nav/meta
        const navItems = [
            { id: 'dashboard', label: 'Dashboard', icon: 'fas fa-gauge-high' },
            { id: 'conversations', label: 'Conversations', icon: 'fas fa-comment-dots' },
            { id: 'database', label: 'Database explorer', icon: 'fas fa-database' },
            { id: 'catalog', label: 'Catalog search', icon: 'fas fa-magnifying-glass-chart' },
            { id: 'clienthistory', label: 'Client history', icon: 'fas fa-clock-rotate-left' },
            { id: 'users', label: 'User management', icon: 'fas fa-users' },
            { id: 'connections', label: 'Connections', icon: 'fas fa-tower-broadcast' },
            { id: 'simulator', label: 'Talk to Kapa', icon: 'fas fa-comments' },
            { id: 'simclients', label: 'Test clients', icon: 'fas fa-user-tag' },
            { id: 'logs', label: 'System logs', icon: 'fas fa-terminal' },
        ];
        const pageTitle = computed(() => navItems.find(i => i.id === currentView.value)?.label || 'Console');

        // ---------------------------------------------------------- loads
        const loadSystem = async () => {
            const res = await fetch('/api/healthz').then(r => r.ok ? r.json() : null).catch(() => null);
            if (res) health.value = res;   // silent: this one polls in the background
        };
        const loadUsers = async () => { const d = await fetchJson('/api/users'); if (d) users.value = d; };
        const loadConnections = async () => { const d = await fetchJson('/api/connections'); if (d) connections.value = d; };
        const loadLogs = async () => {
            let url = `/api/logs?limit=200`;
            if (logsFilter.value.level) url += `&level=${logsFilter.value.level}`;
            if (logsFilter.value.search) url += `&search=${encodeURIComponent(logsFilter.value.search)}`;
            const d = await fetchJson(url);
            if (d) logs.value = d;
        };
        const loadActors = async () => {
            const d = await fetchJson('/api/test/actors');
            if (d?.actors) {
                simActors.value = d.actors;
                if (!sim.value.sender) {
                    const client = simActors.value.find(a => a.type === 'client');
                    if (client) sim.value.sender = client.id;
                }
            }
        };
        const loadModes = async () => { const d = await fetchJson('/api/test/modes'); if (d) simModes.value = d; };

        const loadSimChat = async () => {
            if (!sim.value.sender) return;
            const endpoint = sim.value.channel === 'whatsapp' ? '/api/messages' : '/api/emails';
            const param = sim.value.channel === 'whatsapp'
                ? `phone=${encodeURIComponent(sim.value.sender)}` : `email=${encodeURIComponent(sim.value.sender)}`;
            const data = await fetchJson(`${endpoint}?${param}&limit=50`);
            if (data) simHistory.value = data.reverse();
            const sData = await fetchJson(`/api/test/state/${sim.value.channel}/${encodeURIComponent(sim.value.sender)}`);
            if (sData) simState.value = { ...simState.value, ...sData };
            scrollFeed();
        };

        // ----------------------------------------------------- feed render
        // Persisted history + ephemeral notices, merged and sorted by time.
        const simFeed = computed(() => {
            const items = simHistory.value.map(m => ({
                key: `h${m.id}`,
                ts: new Date(m.timestamp).getTime() || 0,
                mine: m.direction === 'received',            // client side of the chat
                bot: !!m.is_bot,
                text: m.message,
                time: new Date(m.timestamp).toLocaleTimeString(),
                channel: sim.value.channel,
                attachment: m.input_type !== 'text' ? m.input_type : '',
            }));
            for (const n of simNotices.value) items.push(n);
            return items.sort((a, b) => a.ts - b.ts);
        });
        const addNotice = (text, tone = 'dim') => {
            simNotices.value.push({ key: `n${Date.now()}${Math.random()}`, ts: Date.now(), system: true, text, tone });
            if (simNotices.value.length > 20) simNotices.value.shift();
        };
        const scrollFeed = () => nextTick(() => {
            const el = document.getElementById('simChatBox');
            if (el) el.scrollTop = el.scrollHeight;
        });

        // --------------------------------------------------- simulator ops
        const turnPayload = () => ({
            channel: sim.value.channel,
            mode: sim.value.mode,
            sender: sim.value.sender,
            receiver: sim.value.receiver || '',
            text: sim.value.text,
            media_type: sim.value.media_type,
            mock_client_force: sim.value.mock_client_force,
        });

        /** One synchronous round-trip: inject -> flush -> Kapa's real reply. */
        const sendTurn = async () => {
            if (!sim.value.sender) return toast('Pick a client first', 'error');
            if (!sim.value.text && sim.value.media_type === 'text') return;
            loading.value = true;
            const payload = turnPayload();
            sim.value.text = '';
            try {
                const data = await postJson('/api/test/turn', payload);
                if (!data) return;
                if (data.state) simState.value = { ...simState.value, ...data.state, last_benchmark_ms: data.benchmark_ms };
                if (!data.fired) {
                    addNotice('Dropped by the gatekeeper — sender is not a SAGE client (or bot is off for this conversation).', 'bad');
                } else if (!data.replies?.length) {
                    addNotice(simState.value.bot_enabled
                        ? 'Kapa chose silence for this message (<NO_REPLY> or nothing to add).'
                        : 'Client opted out — Kapa stays silent in this conversation.');
                }
                await loadSimChat();   // persisted view: client msg + Kapa's reply
            } finally {
                loading.value = false;
                scrollFeed();
            }
        };

        /** Inject only: the REAL debounce runs (use Flush to fire it early). */
        const sendInjectOnly = async () => {
            if (!sim.value.sender || (!sim.value.text && sim.value.media_type === 'text')) return;
            const payload = turnPayload();
            sim.value.text = '';
            const data = await postJson('/api/test/message', payload);
            if (data) {
                addNotice('Injected. The real debounce is running — press "Fire timer now" to skip the wait.');
                await loadSimChat();
            }
        };

        const flushPending = async () => {
            if (!sim.value.sender) return;
            const receiver = sim.value.receiver ||
                (sim.value.channel === 'whatsapp' ? '34600999001' : 'sales@sim.com');   // Sim Salesman defaults
            loading.value = true;
            try {
                const data = await postJson(
                    `/api/test/flush/${sim.value.channel}/${encodeURIComponent(sim.value.sender)}/${encodeURIComponent(receiver)}`, {});
                if (!data) return;
                if (!data.fired) addNotice('No timer was pending — nothing to fire. This button only '
                    + 'skips the debounce wait for a message sent with "inject only"; it does NOT clear '
                    + 'anything. To erase the conversation use "Clear conversation".');
                await loadSimChat();
            } finally { loading.value = false; }
        };

        const autoGrow = (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
        };

        /** Download the cart as the SAME xlsx the salesman receives on close.
         *  Built server-side by the real _xlsx_from_cart rather than assembled
         *  here: an export that quietly differs from the actual deliverable is
         *  worse than none, because people trust it. Columns: codigo, cantidad. */
        const downloadOrderXlsx = async () => {
            if (!simState.value.confirmed_items?.length) {
                return toast('Cart is empty — nothing to export', 'error');
            }
            const receiver = sim.value.receiver || (sim.value.channel === 'whatsapp'
                ? '34600999001' : 'sales@sim.com');
            const p = new URLSearchParams({ channel: sim.value.channel,
                                            client: sim.value.sender, user: receiver });
            try {
                const res = await fetch(`/api/test/cart/xlsx?${p}`);
                if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
                const url = URL.createObjectURL(await res.blob());
                const link = document.createElement('a');
                link.href = url;
                link.download = `pedido_${sim.value.sender || 'sim'}.xlsx`;
                document.body.appendChild(link); link.click(); document.body.removeChild(link);
                URL.revokeObjectURL(url);
            } catch (e) {
                toast(String(e.message || e), 'error');
            }
        };

        // ------------------------------------------------------ users CRUD
        const users = ref([]);
        const userSearch = ref('');
        const filteredUsers = computed(() => {
            const q = userSearch.value.trim().toLowerCase();
            if (!q) return users.value;
            return users.value.filter(u =>
                [u.name, u.email, u.phone].some(v => (v || '').toLowerCase().includes(q)));
        });
        const newUser = ref({ role: 'user' });
        const showAddUserModal = ref(false);
        const showPairCodeModal = ref(false);
        const pairCode = ref('');
        const showPasswordModal = ref(false);
        const selectedUserForPassword = ref(null);
        const newGmailPassword = ref('');

        const createUser = async () => {
            if (!newUser.value.email || !newUser.value.name) return toast('Name and email are required', 'error');
            const data = await postJson('/api/users', newUser.value);
            if (data) {
                showAddUserModal.value = false;
                newUser.value = { role: 'user' };
                toast('User created');
                loadUsers();
            }
        };
        const deleteUser = async (id) => {
            if (!confirm('Delete this user?')) return;
            const res = await fetchJson(`/api/users/${id}`, { method: 'DELETE' });
            if (res?.success) { toast('User deleted'); loadUsers(); }
        };
        const loginUser = async (user) => {
            loading.value = true;
            const data = await postJson(`/api/users/${user.id}/login`, {});
            loading.value = false;
            if (data?.success) {
                pairCode.value = data.code || '';
                showPairCodeModal.value = true;
                loadUsers();
            } else if (data) {
                toast(`Link failed: ${data.error || 'unknown'}`, 'error');
            }
        };
        const openPasswordModal = (user) => {
            selectedUserForPassword.value = user;
            newGmailPassword.value = '';
            showPasswordModal.value = true;
        };
        const saveGmailPassword = async () => {
            if (!selectedUserForPassword.value || !newGmailPassword.value) return;
            const data = await postJson(
                `/api/users/${selectedUserForPassword.value.id}/gmail-password`,
                { password: newGmailPassword.value });
            if (data?.success) { showPasswordModal.value = false; toast('Password saved'); loadUsers(); }
            else if (data) toast(`Could not save: ${data.error || 'unknown'}`, 'error');
        };

        // -------------------------------------------------------- dashboard
        // ===================================================================
        // INSPECTION VIEWS. Everything below is read-only on the bot's own
        // state, so "why did Kapa do that?" is answerable from the console
        // instead of from app.log. The single destructive action (resetting a
        // conversation) asks for confirmation and is clearly marked.
        // ===================================================================

        // ---------------------------------------------- postgres / db state
        const dbOverview = ref(null);
        const loadDbOverview = async () => {
            const d = await fetchJson('/api/db/overview');
            if (d) dbOverview.value = d;
        };
        const dbTableRows = computed(() => {
            const t = dbOverview.value?.tablas || {};
            const labels = {
                app_users: 'Users',
                chats: 'WhatsApp messages',
                email_chats: 'Email messages',
                conversation_sessions: 'Conversation sessions',
                client_product_history: 'Cached history rows',
                client_history_sync: 'Clients synced',
            };
            return Object.entries(labels).map(([k, label]) => ({
                key: k, label,
                count: t[k],
                error: t[`${k}__error`] || '',
            }));
        });
        // A table is MISSING only when the count came back null (the backend
        // could not query it). Zero is a perfectly healthy empty table, so the
        // advisory must key off this and not merely off "overview loaded".
        const missingTables = computed(() =>
            dbTableRows.value.filter(r => r.count === null || r.count === undefined));

        // -------------------------------------------------- conversations
        const convFilter = ref({ channel: 'all', status: 'all', q: '', solo_abiertas: false });
        const conversations = ref([]);
        const selectedConv = ref(null);
        const convMessages = ref([]);

        const loadConversations = async () => {
            const f = convFilter.value;
            const p = new URLSearchParams({
                channel: f.channel, status: f.status, limit: '200',
                solo_abiertas: f.solo_abiertas ? 'true' : 'false',
            });
            if (f.q.trim()) p.set('q', f.q.trim());
            const d = await fetchJson(`/api/db/sessions?${p}`);
            if (d) conversations.value = d.sesiones || [];
        };

        /** Open one conversation: full session state + its message history. */
        const openConv = async (row) => {
            const d = await fetchJson(`/api/db/sessions/${row.id}`);
            if (!d) return;
            selectedConv.value = d;
            convMessages.value = [];
            const endpoint = d.channel === 'whatsapp' ? '/api/messages' : '/api/emails';
            const param = d.channel === 'whatsapp' ? 'phone' : 'email';
            const msgs = await fetchJson(
                `${endpoint}?${param}=${encodeURIComponent(d.client_id)}&limit=60`);
            if (msgs) convMessages.value = msgs.slice().reverse();
        };

        const resetConv = async (row) => {
            if (!confirm(`Reset the conversation state for ${row.client_id}?\n\n` +
                         'Cart, unresolved queue, memory and the introduction flag are cleared. ' +
                         'Message history is NOT touched. The next client message starts fresh.')) return;
            const d = await fetchJson(`/api/db/sessions/${row.id}`, { method: 'DELETE' });
            if (d?.success) {
                toast('Conversation state reset', 'success');
                selectedConv.value = null;
                loadConversations();
            } else if (d) {
                toast(`Could not reset: ${d.error || 'unknown'}`, 'error');
            }
        };

        const statusTone = (s) => ({
            BUILDING: 'text-warn', CLOSED: 'text-kapa', IDLE: 'text-dim',
        }[s] || 'text-dim');

        const articleName = (code) => {
            const n = selectedConv.value?.nombres_articulos?.[code];
            return n ? `${n.nombre}${n.marca ? ' · ' + n.marca : ''}` : '';
        };

        // -------------------------------------------------- catalog search
        const catalogQuery = ref('');
        const catalogTopK = ref(5);
        const catalogResult = ref(null);

        const searchCatalog = async () => {
            const q = catalogQuery.value.trim();
            if (!q) return;
            loading.value = true;
            try {
                const d = await fetchJson(
                    `/api/db/catalog/search?q=${encodeURIComponent(q)}&top_k=${catalogTopK.value}`);
                if (d) catalogResult.value = d;
            } finally { loading.value = false; }
        };

        const reingestCatalog = async () => {
            if (!confirm('Re-sync the catalog from SAGE into Qdrant?\n\n' +
                         'Incremental (hash diff): cheap when nothing changed, but the FIRST ' +
                         'run embeds everything and can take many minutes.')) return;
            loading.value = true;
            try {
                const d = await postJson('/api/qdrant/ingest', {});
                if (d) { toast(`Ingest done (${d.error || 'ok'})`, 'success'); loadDbOverview(); }
            } finally { loading.value = false; }
        };

        // -------------------------------------------------- client history
        const histIdentifier = ref('');
        const histIdentity = ref(null);
        const histProducts = ref(null);
        const histMentions = ref('');
        const histResolution = ref(null);
        const histClients = ref([]);
        const histShowExcluded = ref(false);

        const mentionList = computed(() =>
            histMentions.value.split('\n').map(s => s.trim()).filter(Boolean));

        const resolveIdentity = async () => {
            const id = histIdentifier.value.trim();
            if (!id) return;
            histProducts.value = null;
            histResolution.value = null;
            const d = await fetchJson(`/api/history/client?identifier=${encodeURIComponent(id)}`);
            if (d) histIdentity.value = d;
        };

        const loadHistProducts = async (refresh = false) => {
            const id = histIdentifier.value.trim();
            if (!id) return;
            loading.value = true;
            try {
                const p = new URLSearchParams({
                    identifier: id, limit: '400',
                    refresh: refresh ? 'true' : 'false',
                    incluir_excluidos: histShowExcluded.value ? 'true' : 'false',
                });
                const d = await fetchJson(`/api/history/products?${p}`);
                if (d) { histProducts.value = d; histIdentity.value = { ...histIdentity.value, ...d }; }
            } finally { loading.value = false; }
        };

        const resolveMentions = async () => {
            const id = histIdentifier.value.trim();
            const list = mentionList.value;
            if (!id || !list.length) return toast('Give an identifier and at least one mention', 'error');
            loading.value = true;
            try {
                const p = new URLSearchParams({ identifier: id });
                list.forEach(m => p.append('q', m));
                const d = await fetchJson(`/api/history/resolve?${p}`);
                if (d) { histResolution.value = d; histIdentity.value = { ...histIdentity.value, ...d }; }
            } finally { loading.value = false; }
        };

        const loadHistClients = async () => {
            const d = await fetchJson('/api/history/clients?limit=200');
            if (d) histClients.value = d.clientes || [];
        };

        /** Verdict colouring, shared by the resolve list and its summary. */
        const verdictTone = (v) => ({
            resolved:  'text-kapa',
            ambiguous: 'text-warn',
            partial:   'text-sales',
            none:      'text-dim',
            disabled:  'text-bad',
        }[v] || 'text-dim');
        const verdictHelp = (v) => ({
            resolved:  'One history article covers every word the client used — safe to commit.',
            ambiguous: 'Several cover everything — this is the case that needs one question.',
            partial:   'Only some words matched. Never committed; falls through to the catalog.',
            none:      'Nothing in this client\'s history shares a term. Honest miss.',
            disabled:  'Ambiguous identity — the history layer is off for this client.',
        }[v] || '');

        // ==================================================== db explorer
        // Browsing shaped around what the tables MEAN: chats render as
        // conversations, orders as a header plus its lines, products as a
        // catalog. The SQL tab is the escape hatch for everything else.
        const dbTab = ref('chats');

        // --- chats
        const chatFilter = ref({ channel: 'all', q: '' });
        const chatThreads = ref([]);
        const openThread = ref(null);
        const loadChatThreads = async () => {
            const p = new URLSearchParams({ channel: chatFilter.value.channel, limit: '300' });
            if (chatFilter.value.q.trim()) p.set('q', chatFilter.value.q.trim());
            const d = await fetchJson(`/api/db/chats/threads?${p}`);
            if (d) chatThreads.value = d.threads || [];
        };
        const openChatThread = async (t) => {
            const p = new URLSearchParams({ channel: t.channel, client: t.client, user: t.user });
            const d = await fetchJson(`/api/db/chats/thread?${p}`);
            if (d) openThread.value = d;
        };

        // --- sage clients
        const sageClientQ = ref('');
        const sageClients = ref([]);
        const loadSageClients = async () => {
            const p = new URLSearchParams({ limit: '100' });
            if (sageClientQ.value.trim()) p.set('q', sageClientQ.value.trim());
            const d = await fetchJson(`/api/sage/clients?${p}`);
            if (d) sageClients.value = d.clientes || [];
        };

        // --- sage orders
        const orderFilter = ref({ codigo_cliente: '', q: '' });
        const sageOrders = ref([]);
        const openOrder = ref(null);
        const loadSageOrders = async () => {
            const p = new URLSearchParams({ limit: '150' });
            if (orderFilter.value.codigo_cliente.trim()) p.set('codigo_cliente', orderFilter.value.codigo_cliente.trim());
            if (orderFilter.value.q.trim()) p.set('q', orderFilter.value.q.trim());
            const d = await fetchJson(`/api/sage/orders?${p}`);
            if (d) { sageOrders.value = d.pedidos || []; openOrder.value = null; }
        };
        const openOrderLines = async (o) => {
            if (openOrder.value?.numero === o.numero && openOrder.value?.ejercicio === o.ejercicio) {
                openOrder.value = null; return;
            }
            const p = new URLSearchParams({ ejercicio: o.ejercicio, numero: o.numero, serie: o.serie || '' });
            const d = await fetchJson(`/api/sage/orders/lines?${p}`);
            if (d) openOrder.value = { ...o, ...d };
        };
        /** From a client row straight to their order history. */
        const ordersForClient = (codigo) => {
            orderFilter.value = { codigo_cliente: codigo, q: '' };
            dbTab.value = 'orders';
            nextTick(() => loadSageOrders());
        };

        // --- catalog
        const prodFilter = ref({ q: '', marca: '', solo_bot: true });
        const sageProducts = ref([]);
        const sageBrands = ref([]);
        const loadSageProducts = async () => {
            const f = prodFilter.value;
            const p = new URLSearchParams({ limit: '200', solo_bot: f.solo_bot ? 'true' : 'false' });
            if (f.q.trim()) p.set('q', f.q.trim());
            if (f.marca) p.set('marca', f.marca);
            const d = await fetchJson(`/api/sage/products?${p}`);
            if (d) sageProducts.value = d.productos || [];
        };
        const loadSageBrands = async () => {
            const d = await fetchJson('/api/sage/brands');
            if (d) sageBrands.value = d.marcas || [];
        };

        // --- raw sql
        const sqlQuery = ref({
            source: 'postgres', limit: 200,
            sql: 'SELECT channel, client_id, order_status, cart\nFROM conversation_sessions\nORDER BY updated_at DESC',
        });
        const sqlResult = ref(null);
        const sqlError = ref('');
        const dbSchema = ref(null);
        const sqlSnippets = [
            { label: 'Bot sessions with a cart', source: 'postgres',
              sql: "SELECT channel, client_id, user_id, order_status, jsonb_array_length(cart::jsonb) AS n\nFROM conversation_sessions\nWHERE cart::text <> '[]'\nORDER BY updated_at DESC" },
            { label: 'Messages per day', source: 'postgres',
              sql: "SELECT date_trunc('day', timestamp) AS dia, count(*) AS n, sum(CASE WHEN is_bot THEN 1 ELSE 0 END) AS del_bot\nFROM chats\nGROUP BY 1 ORDER BY 1 DESC" },
            { label: 'Cached history per client', source: 'postgres',
              sql: 'SELECT codigo_cliente, count(*) AS articulos, sum(veces) AS pedidos\nFROM client_product_history\nGROUP BY 1 ORDER BY 2 DESC' },
            { label: 'SAGE: top products by units', source: 'sage',
              sql: "SELECT TOP 50 l.CodigoArticulo, MAX(l.DescripcionArticulo) AS nombre, SUM(l.UnidadesPedidas) AS uds\nFROM LineasPedidoCliente l\nWHERE l.CodigoEmpresa = 1 AND l.CodigoArticulo <> ''\nGROUP BY l.CodigoArticulo\nORDER BY uds DESC" },
            { label: 'SAGE: most active clients', source: 'sage',
              sql: "SELECT TOP 50 c.CodigoCliente, MAX(c.RazonSocial) AS nombre, COUNT(*) AS pedidos, MAX(c.FechaPedido) AS ultimo\nFROM CabeceraPedidoCliente c\nWHERE c.CodigoEmpresa = 1\nGROUP BY c.CodigoCliente\nORDER BY pedidos DESC" },
        ];
        const applySnippet = (s) => { sqlQuery.value = { ...sqlQuery.value, source: s.source, sql: s.sql }; };

        const runSql = async () => {
            sqlError.value = '';
            loading.value = true;
            try {
                const res = await fetch('/api/db/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sqlQuery.value),
                });
                const d = await res.json();
                if (!res.ok) { sqlError.value = d.detail || 'Query failed'; sqlResult.value = null; }
                else { sqlResult.value = d; }
            } catch (e) {
                sqlError.value = String(e);
            } finally { loading.value = false; }
        };

        const loadSchema = async (source) => {
            const d = await fetchJson(`/api/db/schema?source=${source}`);
            if (d) dbSchema.value = d;
        };

        const loadDbTab = () => {
            if (dbTab.value === 'chats') loadChatThreads();
            if (dbTab.value === 'clients' && !sageClients.value.length) loadSageClients();
            if (dbTab.value === 'orders' && !sageOrders.value.length) loadSageOrders();
            if (dbTab.value === 'products') {
                if (!sageBrands.value.length) loadSageBrands();
                if (!sageProducts.value.length) loadSageProducts();
            }
            if (dbTab.value === 'sql' && !dbSchema.value) loadSchema(sqlQuery.value.source);
        };
        watch(dbTab, loadDbTab);

        const statusCards = computed(() => {
            const details = health.value.details || {};
            const info = (val) => {
                const ok = val === 'ok' || val === true || (Array.isArray(val) && val.length > 0);
                return { color: ok ? 'text-kapa' : 'text-warn',
                         icon: ok ? 'fas fa-circle-check' : 'fas fa-triangle-exclamation',
                         label: ok ? 'OK' : 'CHECK' };
            };
            const pg = info(details.postgres), sql = info(details.sqlserver),
                  wa = info(details.whatsapp_running), qd = info(details.qdrant?.status),
                  ai = info((details.cima || details.cudara)?.status);
            return [
                { label: 'Bot database', desc: 'Users, conversations and sessions (Postgres)', value: pg.label, color: pg.color, icon: 'fas fa-database', statusIcon: pg.icon },
                { label: 'SAGE / ERP', desc: 'Read-only clients and product catalog (SQL Server)', value: sql.label, color: sql.color, icon: 'fas fa-building', statusIcon: sql.icon },
                { label: 'WhatsApp bridge', desc: 'Send/receive engine (meow server)', value: wa.label, color: wa.color, icon: 'fab fa-whatsapp', statusIcon: wa.icon },
                { label: 'Product search', desc: 'Vector index Kapa searches the catalog with (Qdrant)', value: qd.label, color: qd.color, icon: 'fas fa-magnifying-glass', statusIcon: qd.icon },
                { label: 'AI engine', desc: 'The model behind Kapa\'s replies (cima)', value: ai.label, color: ai.color, icon: 'fas fa-brain', statusIcon: ai.icon },
            ];
        });
        const allHealthy = computed(() => statusCards.value.every(c => c.value === 'OK'));

        // -------------------------------------------------- computed (sim)
        const availableReceivers = computed(() => simActors.value.filter(a => ['user', 'admin'].includes(a.type)));
        // Context gauge for the state panel: % of the FULL window, with the
        // trim budget (70% by default) as a visual tick. Colors: teal while
        // comfortably inside the budget, amber approaching it, red beyond.
        const ctxGauge = computed(() => {
            const c = simState.value.ctx || {};
            if (!c.window || !c.used) return null;
            const pct = c.pct_window ?? Math.round(1000 * c.used / c.window) / 10;
            const budgetPct = Math.round(100 * (c.budget || 0) / c.window);
            const over = pct >= budgetPct, near = pct >= budgetPct * 0.85;
            return {
                used: c.used, window: c.window, pct, budgetPct,
                color: over ? 'text-bad' : near ? 'text-warn' : 'text-kapa',
                bar: over ? 'bg-bad' : near ? 'bg-warn' : 'bg-kapa',
            };
        });

        const isNonClientSender = computed(() =>
            !simClients.value.some(c => c.identifier === sim.value.sender || c.key === sim.value.sender)
            && simActors.value.find(a => a.id === sim.value.sender)?.type === 'non_client');
        const modeHelp = computed(() => ({
            logic: 'Everything fake, replies come back instantly through the full pipeline. No devices needed.',
            channel: 'Use your real phone/email as the client. The reply also leaves through the real transport — pick a LINKED receiver.',
        }[sim.value.mode] || ''));

        // -------------------------------------------------------- lifecycle
        let intervals = [];
        const startBackgroundTasks = () => {
            refreshAll();
            intervals.push(setInterval(loadSystem, 8000));
            intervals.push(setInterval(() => { if (currentView.value === 'logs') loadLogs(); }, 4000));
            intervals.push(setInterval(() => { if (currentView.value === 'connections') loadConnections(); }, 6000));
            // Conversations change whenever a debounce fires, so a slow poll
            // keeps the list honest without hammering the DB.
            intervals.push(setInterval(() => {
                if (currentView.value === 'conversations') loadConversations();
            if (currentView.value === 'database') loadDbTab();
            }, 10000));
            // Only channel/full modes need feed polling (replies arrive from
            // real transports); logic mode gets replies synchronously.
            intervals.push(setInterval(() => {
                if (currentView.value === 'simulator' && sim.value.mode !== 'logic') loadSimChat();
            }, 6000));
        };
        onUnmounted(() => intervals.forEach(clearInterval));

        const refreshAll = () => {
            if (!isAuthenticated.value) return;
            loadSystem();
            if (currentView.value === 'dashboard') loadDbOverview();
            if (currentView.value === 'conversations') loadConversations();
            if (currentView.value === 'catalog') loadDbOverview();
            if (currentView.value === 'clienthistory') loadHistClients();
            if (currentView.value === 'users') loadUsers();
            if (currentView.value === 'logs') loadLogs();
            if (currentView.value === 'connections') loadConnections();
            if (currentView.value === 'simulator') { loadActors(); loadModes(); loadSimChat(); loadSimClients(); }
            if (currentView.value === 'simclients') loadSimClients();
        };

        watch(currentView, refreshAll);
        watch(() => [sim.value.sender, sim.value.channel], () => { simNotices.value = []; loadSimChat(); });
        watch(() => [convFilter.value.channel, convFilter.value.status, convFilter.value.solo_abiertas],
              () => { if (currentView.value === 'conversations') loadConversations(); });

        onMounted(() => {
            const expiry = localStorage.getItem(SESSION_KEY);
            if (expiry && Date.now() < parseInt(expiry, 10)) {
                isAuthenticated.value = true;
                localStorage.setItem(SESSION_KEY, Date.now() + SESSION_MINUTES * 60 * 1000);
                startBackgroundTasks();
            } else {
                localStorage.removeItem(SESSION_KEY);
            }
        });

        return {
            // auth
            isAuthenticated, loginUsername, loginPassword, loginError, performLogin,
            // shell
            currentView, navItems, pageTitle, loading, refreshAll, toasts,
            // dashboard
            health, statusCards, allHealthy,
            // db / postgres state
            dbOverview, dbTableRows, missingTables, loadDbOverview,
            // conversations
            convFilter, conversations, selectedConv, convMessages,
            loadConversations, openConv, resetConv, statusTone, articleName,
            // database explorer
            dbTab, loadDbTab,
            chatFilter, chatThreads, openThread, loadChatThreads, openChatThread,
            sageClientQ, sageClients, loadSageClients, ordersForClient,
            orderFilter, sageOrders, openOrder, loadSageOrders, openOrderLines,
            prodFilter, sageProducts, sageBrands, loadSageProducts, loadSageBrands,
            sqlQuery, sqlResult, sqlError, sqlSnippets, applySnippet, runSql,
            dbSchema, loadSchema,
            // catalog search
            catalogQuery, catalogTopK, catalogResult, searchCatalog, reingestCatalog,
            // client history
            histIdentifier, histIdentity, histProducts, histMentions, histResolution,
            histClients, histShowExcluded, mentionList,
            resolveIdentity, loadHistProducts, resolveMentions, loadHistClients,
            verdictTone, verdictHelp,
            // users
            users, filteredUsers, userSearch, newUser, showAddUserModal, createUser, deleteUser, loginUser,
            showPairCodeModal, pairCode, showPasswordModal, selectedUserForPassword, newGmailPassword,
            openPasswordModal, saveGmailPassword,
            // connections + logs
            connections, loadConnections, logs, logsFilter, loadLogs,
            // simulator
            sim, simModes, simActors, simState, simFeed, modeHelp,
            availableReceivers, isNonClientSender, ctxGauge,
            simClients, newSimClient, addSimClient, deleteSimClient, clearSimClients,
            simOrderSeed, simOrders, expandedSimClient,
            seedSimOrders, loadSimOrders, clearSimOrders, inspectSimClient,
            selfChat, selfChatState, seedSelfChat, loadSelfChat, clearSelfChat,
            simReset, resetSimData,
            sendTurn, sendInjectOnly, flushPending, clearSimConversation, loadSimChat, downloadOrderXlsx, autoGrow,
        };
    },
}).mount('#app');