const { createApp, ref, onMounted, computed, watch } = Vue;

createApp({
    setup() {
        // --- Authentication State ---
        const isAuthenticated = ref(false);
        const loginUsername = ref('');
        const loginPassword = ref('');
        const loginError = ref('');

        const SESSION_KEY = 'chatlink_session_expiry';
        const SESSION_MINUTES = 30;

        // Helper to start the intervals
        const startBackgroundTasks = () => {
            refreshAll();
            setInterval(loadSystem, 5000);
            setInterval(() => { if (currentView.value === 'logs') loadLogs(); }, 3000);
            setInterval(() => { if (currentView.value === 'connections') loadConnections(); }, 5000);
        };

        const performLogin = () => {
            if (loginUsername.value === 'admin' && loginPassword.value === 'kapalua') {
                isAuthenticated.value = true;
                loginError.value = '';
                
                // Save expiration time to localStorage
                const expiryTime = Date.now() + (SESSION_MINUTES * 60 * 1000);
                localStorage.setItem(SESSION_KEY, expiryTime);
                
                startBackgroundTasks();
            } else {
                loginError.value = 'Invalid username or password';
            }
        };

        // --- State ---
        const currentView = ref('dashboard');
        const loading = ref(false);
        const health = ref({ status: '...', details: {} });
        
        // Logs
        const logs = ref([]);
        const logsFilter = ref({ level: '', search: '' });

        // Simulator
        const simActors = ref([]);
        const sim = ref({ 
            channel: 'whatsapp', 
            sender: '', 
            receiver: '', 
            text: '', 
            media_type: 'text', 
            force_ai: true 
        });
        const simChat = ref([]);
        const simState = ref({ order_status: '', confirmed_items: [], last_benchmark_ms: 0 });

        // Connections (Monitorization)
        const connections = ref([]);

        // --- Computed for Actor Dropdowns ---
        const availableSenders = computed(() => simActors.value);

        const availableReceivers = computed(() => {
            return simActors.value.filter(a => ['user', 'admin'].includes(a.type));
        });

        const isNonClientSender = computed(() => {
            const actor = simActors.value.find(a => a.id === sim.value.sender);
            return actor && actor.type === 'non_client';
        });

        // --- Navigation ---
        const navItems = [
            { id: 'dashboard', label: 'Dashboard', icon: 'fas fa-tachometer-alt' },
            { id: 'users', label: 'User Management', icon: 'fas fa-users' },
            { id: 'connections', label: 'Active Connections', icon: 'fas fa-network-wired' },
            { id: 'simulator', label: 'Simulator', icon: 'fas fa-flask' },
            { id: 'logs', label: 'System Logs', icon: 'fas fa-terminal' },
        ];
        const pageTitle = computed(() => navItems.find(i => i.id === currentView.value)?.label || 'Console');
        
        // --- Fetch/Load functions ---
        const fetchJson = async (url, opts) => { 
             try {
                const res = await fetch(url, opts);
                if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
                return await res.json();
            } catch (e) { console.error(e); return null; }
        };

        const loadSystem = async () => {
            const h = await fetchJson('/api/healthz');
            if (h) health.value = h;
        };

        const loadUsers = async () => { 
            const data = await fetchJson('/api/users'); 
            if (data) users.value = data; 
        };

        const loadConnections = async () => {
            const data = await fetchJson('/api/connections');
            if (data) connections.value = data;
        };

        const loadLogs = async () => { 
            let url = `/api/logs?limit=200`;
            if (logsFilter.value.level) url += `&level=${logsFilter.value.level}`;
            if (logsFilter.value.search) url += `&search=${encodeURIComponent(logsFilter.value.search)}`;
            const data = await fetchJson(url);
            if (data) logs.value = data;
        };

        const loadActors = async () => {
            const data = await fetchJson('/api/test/actors');
            if (data && data.actors) {
                simActors.value = data.actors;
                if (!sim.value.receiver) {
                    const sales = simActors.value.find(a => a.type === 'user');
                    if (sales) sim.value.receiver = sales.id;
                }
                if (!sim.value.sender) {
                    const client = simActors.value.find(a => a.type === 'client');
                    if (client) sim.value.sender = client.id;
                }
            }
        };

        const loadSimChat = async () => {
            if (!sim.value.sender) return;
            const endpoint = sim.value.channel === 'whatsapp' ? '/api/messages' : '/api/emails';
            const param = sim.value.channel === 'whatsapp' ? `phone=${sim.value.sender}` : `email=${sim.value.sender}`;
            const data = await fetchJson(`${endpoint}?${param}&limit=50`);
            
            if (data) {
                simChat.value = data.reverse();
                setTimeout(() => {
                    const el = document.getElementById('simChatBox');
                    if(el) el.scrollTop = el.scrollHeight;
                }, 50);
            }
            const sData = await fetchJson(`/api/test/state/${sim.value.channel}/${sim.value.sender}`);
            if (sData) simState.value = { ...simState.value, ...sData };
        };

        const sendSimulation = async () => {
            if (!sim.value.sender || !sim.value.receiver) return alert("Sender/Receiver required");
            
            loading.value = true;
            try {
                const payload = { ...sim.value, mock_client_force: false };
                const res = await fetch('/api/test/message', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                simState.value.last_benchmark_ms = data.benchmark_ms || 0;
                sim.value.text = ""; 
                
                let checks = 0;
                const interval = setInterval(async () => {
                    await loadSimChat();
                    checks++;
                    if (checks > 4) { clearInterval(interval); loading.value = false; }
                }, 1000);
            } catch (e) {
                alert("Injection failed");
                loading.value = false;
            }
        };

        const users = ref([]); 
        const filteredUsers = computed(() => users.value);
        const userSearch = ref("");
        const newUser = ref({});
        const showAddUserModal = ref(false);
        const showPairCodeModal = ref(false);
        const pairCode = ref("");
        const showPasswordModal = ref(false);
        const selectedUserForPassword = ref(null);
        const newGmailPassword = ref('');

        const createUser = async () => {
            if (!newUser.value.email || !newUser.value.name) return alert("Datos incompletos");
            const data = await fetchJson('/api/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newUser.value)
            });
            if (data) {
                showAddUserModal.value = false;
                newUser.value = { role: 'user' };
                loadUsers();
            }
        };
        
        const deleteUser = async (id) => {
            if (!confirm("¿Eliminar este usuario?")) return;
            const res = await fetchJson(`/api/users/${id}`, { method: 'DELETE' });
            if (res && res.success) loadUsers();
        };
        
        const loginUser = async (user) => {
            loading.value = true;
            const data = await fetchJson(`/api/users/${user.id}/login`, { method: 'POST' });
            loading.value = false;
            if (data && data.success) {
                pairCode.value = data.code || "";
                showPairCodeModal.value = true;
            } else {
                alert("Error al iniciar sesión: " + (data?.error || "Desconocido"));
            }
        };

        const openPasswordModal = (user) => {
            selectedUserForPassword.value = user;
            newGmailPassword.value = '';
            showPasswordModal.value = true;
        };

        const saveGmailPassword = async () => {
            if (!selectedUserForPassword.value || !newGmailPassword.value) return;
            try {
                const res = await fetch(`/api/users/${selectedUserForPassword.value.id}/gmail-password`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password: newGmailPassword.value })
                });
                const data = await res.json();
                if (data.success) {
                    showPasswordModal.value = false;
                    loadUsers(); // Refresh the table to show the new badge
                } else {
                    alert('Error saving password: ' + data.error);
                }
            } catch (err) {
                console.error(err);
                alert('Connection error');
            }
        };

        const downloadCsv = () => {
            const items = simState.value.confirmed_items;
            if (!items || items.length === 0) {
                alert("No hay productos confirmados para exportar.");
                return;
            }
            const header = "code,qty\n";
            const rows = items.map(item => {
                const code = item.code || item.CodigoArticulo || "N/A";
                const qty = item.qty || 1;
                return `${code},${qty}`;
            }).join("\n");
            const csvContent = "data:text/csv;charset=utf-8," + header + rows;
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `pedido_${sim.value.sender || 'sim'}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };

        const statusCards = computed(() => {
            const details = health.value.details || {};
            
            // Helper to get status color and icon
            const getStatusInfo = (val) => {
                const ok = val === 'ok' || val === true || (Array.isArray(val) && val.length > 0);
                return {
                    color: ok ? 'text-emerald-400' : 'text-amber-400',
                    icon: ok ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle',
                    label: ok ? 'ACTIVO' : 'PROBLEMA'
                };
            };
        
            const pg = getStatusInfo(details.postgres);
            const sql = getStatusInfo(details.sqlserver);
            const wa = getStatusInfo(details.whatsapp_running);
            const qd = getStatusInfo(details.qdrant?.status);
            const ai = getStatusInfo(details.cudara?.status);
        
            return [
                {
                    label: "Base de Datos Principal",
                    desc: "Almacena usuarios y mensajes del bot (Postgres)",
                    value: pg.label,
                    color: pg.color,
                    icon: "fas fa-database",
                    statusIcon: pg.icon
                },
                {
                    label: "Conexión ERP / Clientes",
                    desc: "Lectura de datos de Clientes y Artículos (SQL Server)",
                    value: sql.label,
                    color: sql.color,
                    icon: "fas fa-server",
                    statusIcon: sql.icon
                },
                {
                    label: "Servicio de WhatsApp",
                    desc: "Motor de envío y recepción de mensajes (Meow Server)",
                    value: wa.label,
                    color: wa.color,
                    icon: "fab fa-whatsapp",
                    statusIcon: wa.icon
                },
                {
                    label: "Buscador Inteligente",
                    desc: "Permite al bot buscar productos en el catálogo (Qdrant)",
                    value: qd.label,
                    color: qd.color,
                    icon: "fas fa-search-plus",
                    statusIcon: qd.icon
                },
                {
                    label: "Cerebro Artificial",
                    desc: "Inteligencia que genera las respuestas (AI/Cudara)",
                    value: ai.label,
                    color: ai.color,
                    icon: "fas fa-brain",
                    statusIcon: ai.icon
                }
            ];
        });

        const formatTime = (ts) => new Date(ts).toLocaleTimeString();
        
        // --- Lifecycle ---
        const refreshAll = () => {
            if (!isAuthenticated.value) return; // Guard clause
            loadSystem();
            if (currentView.value === 'users') loadUsers();
            if (currentView.value === 'logs') loadLogs();
            if (currentView.value === 'connections') loadConnections();
            if (currentView.value === 'simulator') {
                loadActors(); 
                loadSimChat();
            }
        };

        watch(currentView, refreshAll);
        watch(() => [sim.value.sender, sim.value.channel], loadSimChat);

        // --- NEW: Check session on page load ---
        onMounted(() => {
            const expiry = localStorage.getItem(SESSION_KEY);
            if (expiry && Date.now() < parseInt(expiry, 10)) {
                // Session is valid. Log them in and extend the timer another 30 mins
                isAuthenticated.value = true;
                localStorage.setItem(SESSION_KEY, Date.now() + (SESSION_MINUTES * 60 * 1000));
                startBackgroundTasks();
            } else {
                // Expired or doesn't exist. Clear it out.
                localStorage.removeItem(SESSION_KEY);
            }
        });

        return {
            // Auth exports
            isAuthenticated, loginUsername, loginPassword, loginError, performLogin,
            
            currentView, navItems, pageTitle, loading, refreshAll,
            health, 
            users, filteredUsers, userSearch, newUser, showAddUserModal, createUser, deleteUser, loginUser,
            connections, loadConnections,
            logs, logsFilter, loadLogs,
            sim, simChat, simState, sendSimulation, loadSimChat, downloadCsv,
            simActors, availableSenders, availableReceivers, isNonClientSender,
            showPairCodeModal, pairCode, formatTime, statusCards
        };
    }
}).mount('#app');