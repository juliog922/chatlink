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
        const newSimClient = ref({ identifier: '', name: '', code: '', notes: '' });
        const loadSimClients = async () => {
            const d = await fetchJson('/api/test/clients');
            if (d) simClients.value = d.clients || [];
        };
        const addSimClient = async () => {
            const c = newSimClient.value;
            if (!c.identifier.trim()) { toast('Identifier (phone or email) is required', 'error'); return; }
            const d = await postJson('/api/test/clients', c);
            if (d && d.ok) {
                toast(`Test client saved: ${d.client.name}`, 'success');
                newSimClient.value = { identifier: '', name: '', code: '', notes: '' };
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
                addNotice('Injected. The real debounce is running — press "Flush pending debounce" to fire it now.');
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
                if (!data.fired) addNotice('Nothing pending to flush for this conversation.');
                await loadSimChat();
            } finally { loading.value = false; }
        };

        const autoGrow = (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
        };

        const downloadCsv = () => {
            const items = simState.value.confirmed_items;
            if (!items?.length) return toast('Cart is empty — nothing to export', 'error');
            const rows = items.map(i => `${i.code || i.CodigoArticulo || 'N/A'},${i.qty || 1}`).join('\n');
            const link = document.createElement('a');
            link.href = encodeURI('data:text/csv;charset=utf-8,code,qty\n' + rows);
            link.download = `pedido_${sim.value.sender || 'sim'}.csv`;
            document.body.appendChild(link); link.click(); document.body.removeChild(link);
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
            if (currentView.value === 'users') loadUsers();
            if (currentView.value === 'logs') loadLogs();
            if (currentView.value === 'connections') loadConnections();
            if (currentView.value === 'simulator') { loadActors(); loadModes(); loadSimChat(); loadSimClients(); }
            if (currentView.value === 'simclients') loadSimClients();
        };

        watch(currentView, refreshAll);
        watch(() => [sim.value.sender, sim.value.channel], () => { simNotices.value = []; loadSimChat(); });

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
            sendTurn, sendInjectOnly, flushPending, loadSimChat, downloadCsv, autoGrow,
        };
    },
}).mount('#app');