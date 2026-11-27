(() => {
  const sidebar = document.getElementById("sidebar");
  const menu = document.getElementById("mobileMenu");
  const backdrop = document.getElementById("sidebarBackdrop");
  const navLinks = Array.from(document.querySelectorAll(".side-nav a"));

  function setSidebarOpen(open) {
    if (sidebar) sidebar.classList.toggle("open", open);
    if (backdrop) backdrop.classList.toggle("open", open);
    if (menu) menu.setAttribute("aria-expanded", String(open));
  }

  menu?.addEventListener("click", () => setSidebarOpen(!sidebar?.classList.contains("open")));
  backdrop?.addEventListener("click", () => setSidebarOpen(false));

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.forEach((item) => {
        item.classList.remove("active");
        item.removeAttribute("aria-current");
      });
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
      setSidebarOpen(false);
    });
  });

  const sections = navLinks
    .map((link) => {
      const href = link.getAttribute("href");
      return href?.startsWith("#") ? document.querySelector(href) : null;
    })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      const matchingLink = navLinks.find((link) => link.getAttribute("href") === `#${visible.target.id}`);
      if (!matchingLink) return;
      navLinks.forEach((link) => {
        link.classList.remove("active");
        link.removeAttribute("aria-current");
      });
      matchingLink.classList.add("active");
      matchingLink.setAttribute("aria-current", "page");
    }, { rootMargin: "-20% 0px -65%", threshold: [0.1, 0.35] });
    sections.forEach((section) => observer.observe(section));
  }

  const search = document.getElementById("userSearch");
  const roleFilter = document.getElementById("roleFilter");
  const statusFilter = document.getElementById("statusFilter");
  const tableBody = document.getElementById("userTableBody");
  const tableWrap = document.getElementById("userTableWrap");
  const tableEmpty = document.getElementById("userTableEmpty");
  const filteredEmpty = document.getElementById("filteredEmpty");
  const countBadge = document.getElementById("userCountBadge");
  let userRequestTimer;

  function addCell(row, label, value) {
    const cell = document.createElement("td");
    cell.dataset.label = label;
    cell.textContent = value || "—";
    row.appendChild(cell);
    return cell;
  }

  function renderUsers(payload) {
    if (!tableBody) return;
    const users = payload.users || [];
    tableBody.replaceChildren();

    users.forEach((user) => {
      const row = document.createElement("tr");
      row.dataset.name = (user.full_name || "").toLowerCase();
      row.dataset.hospital = (user.hospital_name || "").toLowerCase();
      row.dataset.email = (user.email || "").toLowerCase();
      row.dataset.role = user.is_admin ? "admin" : "hospital";
      row.dataset.status = user.is_active ? "active" : "inactive";
      addCell(row, "Full name", user.full_name);
      addCell(row, "Hospital", user.hospital_name);
      addCell(row, "Email", user.email);

      const roleCell = addCell(row, "Role", "");
      const roleBadge = document.createElement("span");
      roleBadge.className = `role-badge ${user.is_admin ? "admin-role" : "hospital-role"}`;
      roleBadge.textContent = user.is_admin ? "Administrator" : "Hospital User";
      roleCell.appendChild(roleBadge);

      const statusCell = addCell(row, "Status", "");
      const statusBadge = document.createElement("span");
      statusBadge.className = `status-badge ${user.is_active ? "active-status" : "inactive-status"}`;
      statusBadge.textContent = user.is_active ? "Active" : "Inactive";
      statusCell.appendChild(statusBadge);

      let registered = "Unknown";
      if (user.created_at) {
        const date = new Date(user.created_at);
        if (!Number.isNaN(date.getTime())) {
          registered = new Intl.DateTimeFormat(undefined, { day: "2-digit", month: "short", year: "numeric" }).format(date);
        }
      }
      addCell(row, "Registered", registered);
      tableBody.appendChild(row);
    });

    const total = Number(payload.total || 0);
    if (tableWrap) tableWrap.hidden = users.length === 0;
    if (tableEmpty) tableEmpty.hidden = total !== 0;
    if (filteredEmpty) filteredEmpty.hidden = total === 0 || users.length !== 0;
    if (countBadge) {
      countBadge.textContent = total > users.length
        ? `Showing ${users.length} of ${total} accounts`
        : `${total} account${total === 1 ? "" : "s"}`;
    }
  }

  async function loadUsers() {
    if (!tableBody) return;
    const params = new URLSearchParams({
      search: search?.value.trim() || "",
      role: roleFilter?.value || "",
      status: statusFilter?.value || "",
      limit: "500",
      offset: "0",
    });
    try {
      const response = await fetch(`/api/admin/users?${params}`, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`User request failed (${response.status})`);
      renderUsers(await response.json());
    } catch (error) {
      console.error("Could not refresh admin user list:", error);
      if (countBadge) countBadge.textContent = "Could not refresh records";
    }
  }

  function scheduleUserLoad() {
    window.clearTimeout(userRequestTimer);
    userRequestTimer = window.setTimeout(loadUsers, 180);
  }
  search?.addEventListener("input", scheduleUserLoad);
  roleFilter?.addEventListener("change", loadUsers);
  statusFilter?.addEventListener("change", loadUsers);

  async function loadSummary() {
    try {
      const response = await fetch("/api/admin/dashboard/summary", { credentials: "same-origin" });
      if (!response.ok) throw new Error(`Summary request failed (${response.status})`);
      const summary = await response.json();
      const values = {
        statTotalUsers: summary.total_users,
        statActiveHospitals: summary.active_hospital_accounts,
        statAdministrators: summary.administrator_accounts,
        statInactiveAccounts: summary.inactive_accounts,
      };
      Object.entries(values).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element && Number.isFinite(Number(value))) element.textContent = String(value);
      });
    } catch (error) {
      console.error("Could not refresh admin statistics:", error);
    }
  }

  function statusClass(value) {
    if (value === "available") return "available";
    if (value === "missing" || value === "unavailable" || value === "invalid") return "unavailable";
    if (value === "stale" || value === "not_configured" || value === "unknown") return "unknown";
    return "pending";
  }

  function setStatus(id, value, label) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = label || String(value || "Unknown").replaceAll("_", " ");
    element.classList.remove("unknown", "pending", "not-connected", "unavailable", "available", "unknown-text", "available-text");
    element.classList.add(statusClass(value));
  }

  function setMonitor(id, dotId, value, label) {
    const element = document.getElementById(id);
    const dot = document.getElementById(dotId);
    if (element) {
      element.textContent = label || String(value || "Unknown").replaceAll("_", " ");
      element.classList.remove("unknown-text", "available-text");
      element.classList.add(value === "available" ? "available-text" : "unknown-text");
    }
    if (dot) {
      dot.classList.remove("unknown-dot", "available-dot");
      dot.classList.add(value === "available" ? "available-dot" : "unknown-dot");
    }
  }

  async function loadSystemStatus() {
    try {
      const response = await fetch("/api/admin/system-status", { credentials: "same-origin" });
      if (!response.ok) throw new Error(`System status request failed (${response.status})`);
      const data = await response.json();
      setStatus("rawDatasetStatus", data.raw_dataset, data.raw_dataset === "not_configured" ? "Not configured" : data.raw_dataset);
      const processedState = data.processed_dataset === "available" && data.data_freshness === "stale"
        ? "stale" : data.processed_dataset;
      const processedLabel = data.processed_dataset === "available"
        ? `Available · ${data.data_freshness}` : data.processed_dataset;
      setStatus("processedDatasetStatus", processedState, processedLabel);
      setStatus("modelTrainingStatus", data.model_artifacts, data.model_artifacts === "available" ? "Artifacts available" : "Unavailable");
      const horizons = data.available_horizons || [];
      setStatus("model2wStatus", horizons.includes(2) ? "available" : "unavailable", horizons.includes(2) ? "Replay available" : "Unavailable");
      setStatus("model4wStatus", horizons.includes(4) ? "available" : "unavailable", horizons.includes(4) ? "Replay available" : "Unavailable");
      setStatus("predictionApiStatus", data.prediction_service, data.prediction_service === "available" ? "Historical replay available" : "Unavailable");
      setMonitor("backendStatus", "backendDot", data.backend, data.backend === "available" ? "Online" : "Unavailable");
      setMonitor("databaseStatus", "databaseDot", data.database, data.database === "available" ? "Connected" : "Unavailable");
      const datasetState = data.processed_dataset === "available" ? (data.data_freshness === "current" ? "available" : "stale") : "unavailable";
      setMonitor("datasetStatus", "datasetDot", datasetState, data.data_through ? `${data.data_freshness}; through ${data.data_through}` : "Unavailable");
      setMonitor("forecastStatus", "forecastDot", data.prediction_service, data.prediction_service === "available" ? "Historical replay ready" : "Unavailable");
      const version = document.getElementById("applicationVersion");
      if (version) version.textContent = data.application_version || "Unknown";
      const badge = document.getElementById("systemStatusBadge");
      if (badge) badge.textContent = data.database === "available" ? "Health checks completed" : "Database unavailable";
      const aiBadge = document.getElementById("aiStatusBadge");
      if (aiBadge) aiBadge.textContent = data.model_artifacts === "available"
        ? `Model artifacts ready · ${horizons.length} horizons`
        : "No compatible model artifacts";
    } catch (error) {
      console.error("Could not load admin system status:", error);
      const badge = document.getElementById("systemStatusBadge");
      if (badge) badge.textContent = "Health check unavailable";
    }
  }

  async function loadActivity() {
    const list = document.getElementById("activityList");
    const empty = document.getElementById("activityEmpty");
    if (!list || !empty) return;
    try {
      const response = await fetch("/api/admin/activity?limit=20", { credentials: "same-origin" });
      if (!response.ok) throw new Error(`Activity request failed (${response.status})`);
      const payload = await response.json();
      list.replaceChildren();
      (payload.events || []).forEach((event) => {
        const article = document.createElement("article");
        article.className = "activity-event";
        const dot = document.createElement("span");
        dot.className = "activity-dot";
        dot.setAttribute("aria-hidden", "true");
        const content = document.createElement("div");
        const description = document.createElement("strong");
        description.textContent = event.description;
        const time = document.createElement("time");
        time.dateTime = event.timestamp;
        const date = new Date(event.timestamp);
        time.textContent = Number.isNaN(date.getTime()) ? event.timestamp : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
        content.append(description, time);
        article.append(dot, content);
        list.appendChild(article);
      });
      const hasEvents = (payload.events || []).length > 0;
      list.hidden = !hasEvents;
      empty.hidden = hasEvents;
    } catch (error) {
      console.error("Could not load admin activity:", error);
      empty.querySelector("p")?.replaceChildren(document.createTextNode("Activity records could not be loaded."));
      empty.hidden = false;
      list.hidden = true;
    }
  }

  const operationsMessage = document.getElementById("operationsAdminMessage");
  let operationsSnapshot = null;

  function appendAdminCell(row, value) {
    const cell = document.createElement("td");
    cell.textContent = value === null || value === undefined || value === "" ? "Not provided" : String(value);
    row.appendChild(cell);
  }

  async function operationsRequest(path, options = {}) {
    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf, ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Operations request failed (${response.status})`);
    return payload;
  }

  function fillSelect(select, values, labelFor) {
    if (!select) return;
    const oldValue = select.value;
    select.replaceChildren(new Option("Select…", ""));
    values.forEach((item) => select.add(new Option(labelFor(item), String(item.value))));
    if (values.some((item) => String(item.value) === oldValue)) select.value = oldValue;
  }

  function fillWardSelect(citySelect, wardSelect) {
    if (!operationsSnapshot || !wardSelect) return;
    const city = citySelect?.value;
    const wards = (operationsSnapshot.wards || []).filter((row) => row.city === city);
    const keep = wardSelect.value;
    wardSelect.replaceChildren(new Option(wardSelect.id === "adminScopeWard" ? "Entire selected city" : "Select ward…", ""));
    wards.forEach((row) => wardSelect.add(new Option(row.ward_id, row.ward_id)));
    if (wards.some((row) => row.ward_id === keep)) wardSelect.value = keep;
  }

  function renderOperations(payload) {
    operationsSnapshot = payload;
    const activeUsers = (payload.hospital_users || []).filter((user) => user.is_active);
    fillSelect(document.getElementById("adminHospitalUser"),
      activeUsers.filter((user) => !user.verified_hospital).map((user) => ({
        value: user.id, user,
      })), (item) => `${item.user.full_name} · ${item.user.hospital_name}`);
    const hospitalNames = [...new Set([
      ...(payload.hospital_inventory || []).map((row) => row.hospital_name),
      ...activeUsers.map((user) => user.verified_hospital).filter(Boolean),
    ])].sort();
    const hospitals = hospitalNames.map((name) => ({ value: name }));
    ["adminHospitalName", "adminCorrectionHospital"].forEach((id) => fillSelect(
      document.getElementById(id), hospitals, (item) => item.value,
    ));
    fillSelect(document.getElementById("adminOperationsUser"), activeUsers.map((user) => ({
      value: user.id, user,
    })), (item) => `${item.user.full_name} · ${item.user.hospital_name}`);
    const cities = [...new Set((payload.wards || []).map((row) => row.city))].sort();
    ["adminWardCity", "adminScopeCity"].forEach((id) => fillSelect(
      document.getElementById(id), cities.map((city) => ({ value: city })), (item) => item.value,
    ));
    fillWardSelect(document.getElementById("adminWardCity"), document.getElementById("adminWardId"));
    fillWardSelect(document.getElementById("adminScopeCity"), document.getElementById("adminScopeWard"));

    const inventoryBody = document.getElementById("adminHospitalInventoryRows");
    inventoryBody?.replaceChildren();
    (payload.hospital_inventory || []).forEach((item) => {
      const row = document.createElement("tr");
      [item.hospital_name, item.actual_available_beds, item.saline_stock_available,
        item.diagnostic_kits_stock_available, item.updated_at].forEach((value) => appendAdminCell(row, value));
      inventoryBody?.appendChild(row);
    });
    const assignmentBody = document.getElementById("adminHospitalWardRows");
    assignmentBody?.replaceChildren();
    (payload.hospital_ward_assignments || []).forEach((item) => {
      const row = document.createElement("tr");
      [item.hospital_name, item.ward_id, item.city].forEach((value) => appendAdminCell(row, value));
      assignmentBody?.appendChild(row);
    });
    const scopeBody = document.getElementById("adminOperationsScopeRows");
    scopeBody?.replaceChildren();
    (payload.operations_scopes || []).forEach((item) => {
      const row = document.createElement("tr");
      [item.user_id, item.city, item.ward_id || "Entire city"].forEach((value) => appendAdminCell(row, value));
      scopeBody?.appendChild(row);
    });
    if (operationsMessage) operationsMessage.textContent =
      `${(payload.wards || []).length} ward references · ${hospitalNames.length} hospitals · ${(payload.operations_scopes || []).length} active scope records.`;
  }

  async function loadOperationsAdmin() {
    if (!operationsMessage) return;
    operationsMessage.textContent = "Loading hospital and ward records…";
    try { renderOperations(await operationsRequest("/api/admin/operations")); }
    catch (error) { operationsMessage.textContent = error.message; }
  }
  document.getElementById("refreshOperations")?.addEventListener("click", loadOperationsAdmin);
  document.getElementById("adminWardCity")?.addEventListener("change", () =>
    fillWardSelect(document.getElementById("adminWardCity"), document.getElementById("adminWardId")));
  document.getElementById("adminScopeCity")?.addEventListener("change", () =>
    fillWardSelect(document.getElementById("adminScopeCity"), document.getElementById("adminScopeWard")));

  document.getElementById("verifyHospitalForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const user = document.getElementById("adminHospitalUser").selectedOptions[0];
    if (!user?.value) return;
    const match = [...(operationsSnapshot?.hospital_users || [])].find((item) => String(item.id) === user.value);
    try {
      await operationsRequest(`/api/admin/hospital-memberships/${user.value}`, {
        method: "PUT", body: JSON.stringify({ hospital_name: match.hospital_name }),
      });
      await loadOperationsAdmin(); loadActivity();
    } catch (error) { operationsMessage.textContent = error.message; }
  });

  document.getElementById("hospitalWardForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const hospital = form.elements.hospital_name.value;
    try {
      await operationsRequest(`/api/admin/hospital-ward-assignments/${encodeURIComponent(hospital)}`, {
        method: "PUT", body: JSON.stringify({ city: form.elements.city.value,
          ward_id: form.elements.ward_id.value, assigned: form.elements.assigned.checked }),
      });
      await loadOperationsAdmin(); loadActivity();
    } catch (error) { operationsMessage.textContent = error.message; }
  });

  document.getElementById("operationsScopeForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    try {
      await operationsRequest(`/api/admin/operations-scopes/${form.elements.user_id.value}`, {
        method: "PUT", body: JSON.stringify({ city: form.elements.city.value,
          ward_id: form.elements.ward_id.value || null, authorized: form.elements.authorized.checked }),
      });
      await loadOperationsAdmin(); loadActivity();
    } catch (error) { operationsMessage.textContent = error.message; }
  });

  document.getElementById("adminInventoryCorrectionForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = { correction_reason: form.elements.correction_reason.value };
    ["actual_available_beds", "saline_stock_available", "diagnostic_kits_stock_available"].forEach((name) => {
      const value = form.elements[name].value.trim();
      if (value !== "") payload[name] = Number(value);
    });
    try {
      await operationsRequest(`/api/admin/hospital-operational-data/${encodeURIComponent(form.elements.hospital_name.value)}`, {
        method: "PUT", body: JSON.stringify(payload),
      });
      form.elements.correction_reason.value = "";
      await loadOperationsAdmin(); loadActivity();
    } catch (error) { operationsMessage.textContent = error.message; }
  });

  loadUsers();
  loadSummary();
  loadSystemStatus();
  loadActivity();
  loadOperationsAdmin();

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setSidebarOpen(false);
  });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 1050) setSidebarOpen(false);
  });
})();
