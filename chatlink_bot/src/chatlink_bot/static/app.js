const { createApp, ref, onMounted, computed, watch } = Vue;

createApp({
    setup() {
        // --- State ---
        const currentView = ref('dashboard');
        const loading = ref(false);
        const health = ref({ status: '...', details: {} });
        const resources = ref({ ram: { used_mb: 0, total_mb: 0 }, gpu: { available: false, gpus: [] } });
        const config = ref({});
        
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

        // --- Computed for Actor Dropdowns ---
        const availableSenders = computed(() => {
            // Senders can be anyone (Clients, Non-Clients, or Users acting as senders)
            // But typically for testing BOT logic, Sender = Client/Non-Client
            return simActors.value; 
        });

        const availableReceivers = computed(() => {
            // Receivers must be internal (Salesman/Admin)
            return simActors.value.filter(a => ['user', 'admin'].includes(a.type));
        });

        // Detect if the selected Sender is a "Non-Client"
        const isNonClientSender = computed(() => {
            const actor = simActors.value.find(a => a.id === sim.value.sender);
            return actor && actor.type === 'non_client';
        });

        // ... (Nav items, existing computed props) ...
        const navItems = [
            { id: 'dashboard', label: 'Dashboard', icon: 'fas fa-tachometer-alt' },
            { id: 'users', label: 'User Management', icon: 'fas fa-users' },
            { id: 'simulator', label: 'Simulator', icon: 'fas fa-flask' },
            { id: 'logs', label: 'System Logs', icon: 'fas fa-terminal' },
        ];
        const pageTitle = computed(() => navItems.find(i => i.id === currentView.value)?.label || 'Console');
        // ... (statusCards, filteredUsers, etc. - keep existing) ...
        
        // ... (Fetch/Load functions) ...
        const fetchJson = async (url, opts) => { /* keep existing */ 
             try {
                const res = await fetch(url, opts);
                if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
                return await res.json();
            } catch (e) { console.error(e); return null; }
        };

        const loadSystem = async () => {
            const [h, r, c] = await Promise.all([
                fetchJson('/api/healthz'),
                fetchJson('/api/system/resources'),
                fetchJson('/api/system/config')
            ]);
        
            if (h) health.value = h;
            if (r) resources.value = r;
            if (c) config.value = c;
        };
        const loadUsers = async () => { 
            const data = await fetchJson('/api/users'); 
            if (data) users.value = data; // Note: Ensure `users` ref is defined (it was in original)
        };
        const loadLogs = async () => { /* keep existing */ 
            let url = `/api/logs?limit=200`;
            if (logsFilter.value.level) url += `&level=${logsFilter.value.level}`;
            if (logsFilter.value.search) url += `&search=${encodeURIComponent(logsFilter.value.search)}`;
            const data = await fetchJson(url);
            if (data) logs.value = data;
        };

        // NEW: Load Actors
        const loadActors = async () => {
            const data = await fetchJson('/api/test/actors');
            if (data && data.actors) {
                simActors.value = data.actors;
                // Defaults
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
            // Use user OR client search
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
                // If it is a non-client, we do NOT force mock. 
                // If it is a client/user, we FORCE mock (in case SQL server is inconsistent in dev).
                // Actually, logic: If actor.type == 'non_client', mock_client_force = false.
                // Else mock_client_force = true (to be safe in simulation).
                // BUT per user request: "Common (client) ... we check the sqlserver".
                // So if we pick a 'client' type, we expect it to exist. We should NOT force mock.
                // Only force mock if we are testing a hypothetical client that doesn't exist in DB.
                // To keep it simple: Let's assume we rely on real DB data for 'client' types, 
                // and we rely on DB absence for 'non_client'.
                // So mock_client_force = false always, unless we add a specific "Hypothetical Client" option.
                
                const payload = {
                    ...sim.value,
                    mock_client_force: false // Rely on DB logic (Success for Client, Fail for Non-Client)
                };

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

        // ... (Keep other actions: createUser, deleteUser, loginUser, downloadCsv, formatTime) ...
        const users = ref([]); // Ensure users is defined
        const filteredUsers = computed(() => users.value); // simple stub
        const userSearch = ref("");
        const newUser = ref({});
        const showAddUserModal = ref(false);
        const showQrModal = ref(false);
        const qrData = ref("");

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
                qrData.value = data.qr || ""; // Datos para el código QR
                showQrModal.value = true;
            } else {
                alert("Error al iniciar sesión: " + (data?.error || "Desconocido"));
            }
        };
        const downloadCsv = () => {
            const items = simState.value.confirmed_items;
            if (!items || items.length === 0) {
                alert("No hay productos confirmados para exportar.");
                return;
            }
        
            // Crear cabecera y filas
            const header = "code,qty\n";
            const rows = items.map(item => {
                const code = item.code || item.CodigoArticulo || "N/A";
                const qty = item.qty || 1;
                return `${code},${qty}`;
            }).join("\n");
        
            const csvContent = "data:text/csv;charset=utf-8," + header + rows;
            const encodedUri = encodeURI(csvContent);
            
            // Crear link temporal para la descarga
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `pedido_${sim.value.sender || 'sim'}.csv`);
            document.body.appendChild(link);
            
            link.click(); // Disparar descarga
            document.body.removeChild(link);
        };
        const statusCards = computed(() => ({
            health: {
                label: "Estado Global",
                value: health.value.status.toUpperCase(),
                icon: "fas fa-heartbeat",
                statusIcon: health.value.status === 'ok' ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle',
                color: health.value.status === 'ok' ? 'text-emerald-400' : 'text-amber-400',
                subtext: `Cudara: ${health.value.details?.cudara?.status || '?'}`
            },
            gpu: {
                label: "Hardware AI",
                value: resources.value.gpu?.available ? 'GPU ACTIVA' : 'CPU MODE',
                icon: "fas fa-microchip",
                statusIcon: resources.value.gpu?.available ? 'fas fa-bolt' : 'fas fa-server',
                color: resources.value.gpu?.available ? 'text-purple-400' : 'text-slate-400',
                subtext: resources.value.gpu?.available ? resources.value.gpu.gpus[0].name : 'Sin aceleración'
            }
            // Puedes añadir más según necesites
        }));

        const formatTime = (ts) => new Date(ts).toLocaleTimeString();
        
        // --- Lifecycle ---
        const refreshAll = () => {
            loadSystem();
            if (currentView.value === 'users') loadUsers();
            if (currentView.value === 'logs') loadLogs();
            if (currentView.value === 'simulator') {
                loadActors(); // New
                loadSimChat();
            }
        };

        watch(currentView, refreshAll);
        watch(() => [sim.value.sender, sim.value.channel], loadSimChat);

        onMounted(() => {
            refreshAll();
            setInterval(loadSystem, 5000);
            setInterval(() => { if (currentView.value === 'logs') loadLogs(); }, 3000);
        });

        return {
            currentView, navItems, pageTitle, loading, refreshAll,
            health, resources, config, 
            // Users
            users, filteredUsers, userSearch, newUser, showAddUserModal, createUser, deleteUser, loginUser,
            // Logs
            logs, logsFilter, loadLogs,
            // Sim
            sim, simChat, simState, sendSimulation, loadSimChat, downloadCsv,
            simActors, availableSenders, availableReceivers, isNonClientSender,
            // Modal
            showQrModal, qrData, formatTime, 
            statusCards: computed(() => { /* keep original statusCards logic */ return {}; }) 
        };
    }
}).mount('#app');