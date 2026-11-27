"use strict";

const sidebar = document.getElementById("sidebar");
const menuButton = document.getElementById("mobileMenu");
const backdrop = document.getElementById("sidebarBackdrop");
const mobileViewport = window.matchMedia("(max-width: 900px)");

function setMenuOpen(isOpen, returnFocus = false) {
    const shouldOpen = isOpen && mobileViewport.matches;

    sidebar.classList.toggle("open", shouldOpen);
    backdrop.classList.toggle("is-visible", shouldOpen);
    menuButton.setAttribute("aria-expanded", String(shouldOpen));
    menuButton.setAttribute(
        "aria-label",
        shouldOpen ? "Close navigation menu" : "Open navigation menu"
    );
    document.body.classList.toggle("nav-open", shouldOpen);

    if (shouldOpen) {
        sidebar.querySelector("a")?.focus();
    } else if (returnFocus) {
        menuButton.focus();
    }
}

menuButton.addEventListener("click", () => {
    setMenuOpen(menuButton.getAttribute("aria-expanded") !== "true");
});
backdrop.addEventListener("click", () => setMenuOpen(false));

sidebar.querySelectorAll("nav a").forEach((link) => {
    link.addEventListener("click", () => {
        sidebar.querySelectorAll("nav a").forEach((item) => {
            item.classList.remove("active");
            item.removeAttribute("aria-current");
        });

        link.classList.add("active");
        link.setAttribute("aria-current", "location");
        setMenuOpen(false);
    });
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
        setMenuOpen(false, true);
    }
});

mobileViewport.addEventListener("change", (event) => {
    if (!event.matches) {
        setMenuOpen(false);
    }
});

const forecastForm = document.getElementById("forecastForm");
const citySelect = document.getElementById("citySelect");
const wardSelect = document.getElementById("wardSelect");
const generateButton = document.getElementById("generateForecast");
const buttonLabel = document.getElementById("buttonLabel");
const buttonSpinner = document.getElementById("buttonSpinner");
const forecastResults = document.getElementById("forecast-results");
const forecastMessage = document.getElementById("forecastMessage");
const resultBadge = document.getElementById("resultBadge");
const forecastEmpty = document.getElementById("forecastEmpty");
const forecastLoading = document.getElementById("forecastLoading");
const forecastError = document.getElementById("forecastError");
const forecastErrorMessage = document.getElementById("forecastErrorMessage");
const forecastSuccess = document.getElementById("forecastSuccess");
const driversEmpty = document.getElementById("driversEmpty");
const driverList = document.getElementById("driverList");
const chartEmpty = document.getElementById("chartEmpty");
const chartContent = document.getElementById("chartContent");
const asOfDate = document.getElementById("asOfDate");
const currentMode = document.getElementById("currentMode");
let requestSequence = 0;
let activeForecastController = null;
const FORECAST_TIMEOUT_MS = 30000;

function availableHorizons() {
    try {
        return JSON.parse(forecastForm.dataset.availableHorizons || "[]").map(Number);
    } catch (_) {
        return [];
    }
}

function dateBounds() {
    try {
        return JSON.parse(forecastForm.dataset.dateBounds || "{}");
    } catch (_) {
        return {};
    }
}

function selectedHorizon() {
    return Number(forecastForm.querySelector('input[name="horizon"]:checked')?.value);
}

function updateDateBounds() {
    const allBounds = dateBounds();
    const areaBounds = allBounds.areas?.[citySelect.value]?.[wardSelect.value];
    const bounds = areaBounds || { min: allBounds.min, max: allBounds.max };
    if (!bounds.min || !bounds.max) return;
    asOfDate.min = bounds.min;
    asOfDate.max = bounds.max;
    if (!asOfDate.value || asOfDate.value < bounds.min || asOfDate.value > bounds.max) {
        asOfDate.value = bounds.max;
    }
}

function refreshControlAvailability() {
    const available = availableHorizons();
    forecastForm.querySelectorAll('input[name="horizon"]').forEach((input) => {
        input.disabled = !available.includes(Number(input.value));
    });
    citySelect.disabled = !citySelect.options.length || citySelect.options.length <= 1;
    wardSelect.disabled = !citySelect.value || wardSelect.options.length <= 1;
    currentMode.disabled = currentMode.dataset.available !== "true";
    asOfDate.disabled = currentMode.checked || available.length === 0;
    asOfDate.required = !currentMode.checked;
    generateButton.disabled = forecastResults.getAttribute("aria-busy") === "true"
        || available.length === 0 || !citySelect.value || !wardSelect.value;
}

function populateWards() {
    const cityOption = citySelect.selectedOptions[0];
    let wards = [];
    try {
        wards = cityOption?.dataset.wards ? JSON.parse(cityOption.dataset.wards) : [];
    } catch (_) {
        wards = [];
    }

    wardSelect.replaceChildren(new Option(wards.length ? "Select a ward" : "Select a city first", ""));
    wardSelect.disabled = wards.length === 0;

    wards.forEach((wardId) => {
        const label = `Ward ${wardId.split("_Ward_").at(-1)}`;
        wardSelect.add(new Option(label, wardId));
    });
}

function setForecastState(state, message = "") {
    const isLoading = state === "loading";
    const horizonInputs = forecastForm.querySelectorAll('input[name="horizon"]');

    forecastResults.setAttribute("aria-busy", String(isLoading));
    forecastEmpty.hidden = state !== "empty";
    forecastLoading.hidden = !isLoading;
    forecastError.hidden = state !== "error";
    forecastSuccess.hidden = state !== "success";
    driversEmpty.hidden = state === "success";
    driverList.hidden = state !== "success";
    chartEmpty.hidden = state === "success";
    chartContent.hidden = state !== "success";
    refreshControlAvailability();
    buttonSpinner.hidden = !isLoading;
    buttonLabel.textContent = isLoading ? "Preparing forecast" : "Generate forecast";

    if (state === "loading") {
        resultBadge.textContent = "Preparing";
        forecastMessage.textContent = "Preparing your selected forecast.";
    } else if (state === "error") {
        resultBadge.textContent = "Unavailable";
        forecastMessage.textContent = "No prediction was generated.";
        forecastErrorMessage.textContent = message;
    } else if (state === "success") {
        resultBadge.textContent = "Forecast ready";
        forecastMessage.textContent = "Forecast results are ready.";
    } else {
        resultBadge.textContent = "Awaiting forecast";
        forecastMessage.textContent = "Select a city and ward to prepare your forecast request.";
        forecastErrorMessage.textContent = "";
    }
}

function clearForecast() {
    setForecastState("empty");
}

function invalidatePendingForecast() {
    requestSequence += 1;
    if (activeForecastController) {
        activeForecastController.abort();
        activeForecastController = null;
    }
}

citySelect.addEventListener("change", () => {
    invalidatePendingForecast();
    populateWards();
    updateDateBounds();
    clearForecast();
});

wardSelect.addEventListener("change", () => {
    invalidatePendingForecast();
    updateDateBounds();
    clearForecast();
});
forecastForm.querySelectorAll('input[name="horizon"]').forEach((input) => {
    input.addEventListener("change", () => {
        invalidatePendingForecast();
        clearForecast();
    });
});
currentMode.addEventListener("change", () => {
    invalidatePendingForecast();
    refreshControlAvailability();
    clearForecast();
});

function formatDate(value) {
    if (typeof value !== "string" || Number.isNaN(Date.parse(value))) {
        throw new Error("The forecast service returned an invalid observation date.");
    }

    return new Intl.DateTimeFormat("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
    }).format(new Date(value));
}

function validateSeries(series, name, minimumLength) {
    if (
        !Array.isArray(series)
        || series.length < minimumLength
        || series.some((point) => (
            !point
            || typeof point.date !== "string"
            || Number.isNaN(Date.parse(point.date))
            || !Number.isFinite(point.cases)
            || point.cases < 0
        ))
    ) {
        throw new Error(`The forecast service returned invalid ${name} data.`);
    }

    return series;
}

function drawForecastChart(history, forecast, selection) {
    const svgNamespace = "http://www.w3.org/2000/svg";
    const chartGrid = document.getElementById("chartGrid");
    const chartYAxis = document.getElementById("chartYAxis");
    const historyLine = document.getElementById("historyLine");
    const forecastLine = document.getElementById("forecastLine");
    const divider = document.getElementById("forecastDivider");
    const chartDescription = document.getElementById("chartSvgDescription");
    const chartTitle = document.getElementById("chartSvgTitle");
    const allValues = [...history, ...forecast].map((point) => point.cases);
    const maxCases = Math.max(1, ...allValues);
    const plot = { left: 58, right: 736, top: 20, bottom: 236 };
    const totalPoints = history.length + forecast.length;

    chartGrid.replaceChildren();
    chartYAxis.replaceChildren();

    for (let line = 0; line < 5; line += 1) {
        const y = plot.top + ((plot.bottom - plot.top) * line) / 4;
        const gridLine = document.createElementNS(svgNamespace, "line");
        gridLine.setAttribute("x1", String(plot.left));
        gridLine.setAttribute("x2", String(plot.right));
        gridLine.setAttribute("y1", String(y));
        gridLine.setAttribute("y2", String(y));
        chartGrid.append(gridLine);

        const axisLabel = document.createElementNS(svgNamespace, "text");
        axisLabel.setAttribute("x", String(plot.left - 10));
        axisLabel.setAttribute("y", String(y + 4));
        axisLabel.setAttribute("text-anchor", "end");
        axisLabel.textContent = new Intl.NumberFormat(undefined, {
            maximumFractionDigits: 1,
        }).format(maxCases * (4 - line) / 4);
        chartYAxis.append(axisLabel);
    }

    function pointCoordinates(point, index) {
        const x = plot.left + ((plot.right - plot.left) * index) / (totalPoints - 1);
        const y = plot.bottom - ((plot.bottom - plot.top) * point.cases) / maxCases;
        return `${x},${y}`;
    }

    historyLine.setAttribute(
        "points",
        history.map(pointCoordinates).join(" ")
    );
    forecastLine.setAttribute(
        "points",
        [history[history.length - 1], ...forecast]
            .map((point, index) => pointCoordinates(
                point,
                history.length - 1 + index
            ))
            .join(" ")
    );

    const dividerX = plot.left
        + ((plot.right - plot.left) * (history.length - 1)) / (totalPoints - 1);
    divider.setAttribute("x1", String(dividerX));
    divider.setAttribute("x2", String(dividerX));
    divider.setAttribute("y1", String(plot.top));
    divider.setAttribute("y2", String(plot.bottom));

    chartTitle.textContent = `${selection.city} ${wardSelect.selectedOptions[0].textContent}: observed weekly new cases and forecast`;
    chartDescription.textContent = `Observed new dengue cases for each reporting week through ${formatDate(history.at(-1).date)}, followed by a forecast for the target reporting week.`;
    document.getElementById("chartStartDate").textContent = formatDate(history[0].date);
    document.getElementById("chartObservedDate").textContent = formatDate(history.at(-1).date);
    document.getElementById("chartForecastDate").textContent = formatDate(forecast.at(-1).date);
}

function renderForecastResult(result, selection) {
    const expectedPrediction = Number.isFinite(result?.expectedPredictedCases)
        ? result.expectedPredictedCases
        : Math.max(0, result?.rawPredictedCases);
    if (
        !result
        || !Number.isFinite(result.rawPredictedCases)
        || !Number.isFinite(expectedPrediction)
        || expectedPrediction < 0
        || !Number.isFinite(result.predictedCases)
        || result.predictedCases < 0
        || !Number.isInteger(result.predictedCases)
        || typeof result.targetDate !== "string"
        || typeof result.observationDate !== "string"
        || typeof result.riskCategory !== "string"
        || !result.riskCategory.trim()
        || !result.drivers
    ) {
        throw new Error("The forecast service returned an incomplete result.");
    }

    const history = validateSeries(result.history, "historical case", 2);
    const forecast = validateSeries(result.forecast, "forecast", 1);
    if (forecast.length !== 1 || forecast[0].date !== result.targetDate
        || result.horizonWeeks !== selection.horizonWeeks) {
        throw new Error("The forecast response does not match the selected horizon and target week.");
    }
    if (selection.asOf) {
        const requested = Date.parse(`${selection.asOf}T00:00:00Z`);
        const resolved = Date.parse(`${result.observationDate}T00:00:00Z`);
        const target = Date.parse(`${result.targetDate}T00:00:00Z`);
        if (result.requestedAsOf !== selection.asOf || resolved > requested
            || requested - resolved >= 7 * 24 * 60 * 60 * 1000
            || target - resolved !== selection.horizonWeeks * 7 * 24 * 60 * 60 * 1000) {
            throw new Error("The selected calendar date did not resolve to the expected weekly forecast origin.");
        }
    }
    const drivers = [
        ["rainfallDriver", result.drivers.rainfall],
        ["temperatureDriver", result.drivers.temperature],
        ["humidityDriver", result.drivers.humidity],
        ["recentCasesDriver", result.drivers.recentCases],
    ];

    if (drivers.some(([, value]) => typeof value !== "string" || !value.trim())) {
        throw new Error("The forecast service returned incomplete driver explanations.");
    }

    document.getElementById("expectedPredictedCases").textContent =
        new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
            .format(expectedPrediction);
    document.getElementById("roundedPredictedCases").textContent =
        `Rounded case estimate: ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })
            .format(result.predictedCases)} cases`;
    const rawModelOutput = document.getElementById("rawModelOutput");
    rawModelOutput.hidden = !(result.rawPredictedCases < 0 && !result.rawPredictionIsClipped);
    rawModelOutput.textContent = rawModelOutput.hidden
        ? ""
        : `Raw model output before the non-negative case-count constraint: ${new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(result.rawPredictedCases)} cases`;
    document.getElementById("riskCategory").textContent = result.riskCategory;
    document.getElementById("riskCategory").dataset.risk = result.riskCategory;
    document.getElementById("predictionPeriod").textContent =
        `${selection.horizonWeeks}-week horizon · target reporting week: ${formatDate(result.targetDate)}`;
    document.getElementById("predictionModel").textContent =
        `Selected model: ${result.model} · ${selection.horizonWeeks}-week horizon · backend: ${result.predictionBackend || "phase2"}`;
    const forecastKind = result.isHistoricalReplay
        ? "Leakage-safe held-out backtest replay"
        : result.isRetrospectiveEstimate
            ? "Retrospective historical estimate"
            : selection.asOf
                ? "Historical estimate"
                : "Current forecast";
    const requestedDateText = result.requestedAsOf
        ? `requested date: ${formatDate(result.requestedAsOf)} · `
        : "";
    document.getElementById("observationDate").textContent =
        `${forecastKind} · ${selection.city}, ${wardSelect.selectedOptions[0].textContent} · ${requestedDateText}resolved observation: ${formatDate(result.observationDate)} · target week: ${formatDate(result.targetDate)} · model training data through: ${formatDate(result.modelTrainingDataThrough || result.modelTrainingDataCutoff)} · ${result.model}`;
    const riskWarning = document.getElementById("riskWarning");
    riskWarning.textContent = [result.predictionWarning, result.riskWarning, ...(result.warnings || []).filter((warning) => warning !== result.riskWarning && warning !== result.predictionWarning)].filter(Boolean).join(" ")
        || "Experimental ward-relative classifications; not clinically validated public-health alerts.";
    riskWarning.hidden = false;
    if (result.actualCases !== null && Number.isFinite(result.actualCases)) {
        document.getElementById("observationDate").textContent +=
            ` · observed target-week count: ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(result.actualCases)}`;
    } else {
        document.getElementById("observationDate").textContent += " · target-week actual is not present in the dataset";
    }

    drivers.forEach(([elementId, value]) => {
        document.getElementById(elementId).textContent = value;
    });

    const featureRows = [
        ["featureAvgTemp", "avg_temp_C", "°C"],
        ["featureHumidity", "relative_humidity_pct", "%"],
        ["featureRainfall", "rainfall_mm", " mm"],
        ["featureRainLag1", "rain_lag_1w", " mm"],
        ["featureRainLag2", "rain_lag_2w", " mm"],
        ["featureRainLag3", "rain_lag_3w", " mm"],
        ["featureRainLag4", "rain_lag_4w", " mm"],
        ["featureRainSum2", "rain_sum_2w", " mm"],
        ["featureRainSum4", "rain_sum_4w", " mm"],
        ["featureDensity", "population_density_sqkm", " people/km²"],
        ["featureWaterlogging", "water_logging_risk_index", ""],
        ["featureSanitation", "sanitation_score", ""],
    ];
    const observedFeatures = result.observedFeatures || {};
    featureRows.forEach(([elementId, featureName, unit]) => {
        const value = observedFeatures[featureName];
        document.getElementById(elementId).textContent = Number.isFinite(value)
            ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value)}${unit}`
            : "Unavailable";
    });
    document.getElementById("observedFeatureList").hidden = false;

    const rainfallTiming = result.rainfallTiming?.target_minus_three_weeks;
    const rainfallTimingSummary = document.getElementById("rainfallTimingSummary");
    rainfallTimingSummary.textContent = rainfallTiming
        ? `Rainfall observed ${formatDate(rainfallTiming.date)} (${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(rainfallTiming.rainfall_mm)} mm) is approximately three weeks before the target reporting week. This is timing context, not evidence of causation.`
        : "Rainfall approximately three weeks before the target week would occur after this forecast origin and is unavailable. No future rainfall observation was used.";
    rainfallTimingSummary.hidden = false;

    renderModelPerformance(result.modelPerformance, selection.horizonWeeks);

    drawForecastChart(history, forecast, selection);
    setForecastState("success");
}

function renderModelPerformance(performance, horizon) {
    const empty = document.getElementById("performanceEmpty");
    const grid = document.getElementById("performanceGrid");
    const note = document.getElementById("performanceNote");
    if (!performance || typeof performance !== "object") {
        empty.hidden = false;
        grid.hidden = true;
        note.hidden = true;
        return;
    }
    const number = (value) => Number.isFinite(value)
        ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)
        : "Unavailable";
    const percent = (value) => Number.isFinite(value)
        ? new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(value)
        : "Unavailable";
    const fields = [
        ["validationMae", performance.validationMae],
        ["validationRmse", performance.validationRmse],
        ["testMae", performance.testMae],
        ["persistenceMae", performance.persistenceMae],
        ["seasonalMae", performance.seasonalNaiveMae],
        ["testR2", performance.testR2],
    ];
    fields.forEach(([id, value]) => { document.getElementById(id).textContent = number(value); });
    const risk = performance.risk || {};
    document.getElementById("riskExact").textContent = percent(risk.exact_tier_accuracy);
    document.getElementById("riskRecall").textContent = percent(risk.high_critical_recall);
    document.getElementById("criticalRecall").textContent = percent(risk.critical_recall);
    empty.hidden = true;
    grid.hidden = false;
    note.hidden = false;
    note.textContent = `${performance.model} · ${horizon}-week model · validation MAE ${number(performance.validationMae)} · test origins ${formatDate(performance.originStart)} to ${formatDate(performance.originEnd)}; targets ${formatDate(performance.targetStart)} to ${formatDate(performance.targetEnd)}. Historical rolling-origin estimates only.`;
}

async function requestForecast(selection, signal) {
    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
    const response = await fetch("/api/predict", {
        method: "POST",
        credentials: "same-origin",
        signal,
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({
            city: selection.city,
            ward: selection.ward,
            horizonWeeks: selection.horizonWeeks,
            asOf: selection.asOf || null,
        }),
    });
    let data = {};
    try {
        data = await response.json();
    } catch (_) {
        // A non-JSON server response is rendered as a generic service error.
    }
    if (!response.ok) {
        if (response.status === 403
            && data.detail === "Invalid or missing CSRF token.") {
            throw new Error(
                "Your dashboard session expired or changed. Refresh this page, sign in again if prompted, then submit the forecast again."
            );
        }
        const detail = typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
                ? data.detail.map((item) => item.msg || "Invalid request").join("; ")
                : `Forecast request failed (${response.status}).`;
        throw new Error(detail);
    }
    return data;
}

forecastForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!forecastForm.reportValidity()
        || !citySelect.value || !wardSelect.value
        || !availableHorizons().includes(selectedHorizon())) {
        return;
    }

    const selection = {
        city: citySelect.value,
        ward: wardSelect.value,
        horizonWeeks: Number(
            forecastForm.querySelector('input[name="horizon"]:checked').value
        ),
        asOf: currentMode.checked ? null : asOfDate.value,
    };

    if (activeForecastController) activeForecastController.abort();
    const controller = new AbortController();
    activeForecastController = controller;
    const sequence = ++requestSequence;
    const timeout = window.setTimeout(() => controller.abort(), FORECAST_TIMEOUT_MS);
    setForecastState("loading");

    try {
        const result = await requestForecast(selection, controller.signal);
        if (sequence !== requestSequence) return;
        renderForecastResult(result, selection);
    } catch (error) {
        if (sequence !== requestSequence) return;
        const message = controller.signal.aborted
            ? "The forecast request timed out after 30 seconds. Your division and ward selections are available; retry the forecast."
            : error instanceof Error
                ? error.message
                : "The forecast could not be generated. No prediction was returned.";
        setForecastState("error", message);
    } finally {
        window.clearTimeout(timeout);
        if (activeForecastController === controller) activeForecastController = null;
    }
});

currentMode.dataset.available = currentMode.disabled ? "false" : "true";
if (!forecastForm.querySelector('input[name="horizon"]:checked')
    || !availableHorizons().includes(selectedHorizon())) {
    const firstAvailable = forecastForm.querySelector(`input[name="horizon"][value="${availableHorizons()[0]}"]`);
    if (firstAvailable) firstAvailable.checked = true;
}
populateWards();
updateDateBounds();
refreshControlAvailability();

// Ward vulnerability, capacity, hospital inventory, operations, allocation,
// and scenario tools use the operational reference data independently of ML.
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";
const vulnerabilityCityFilter = document.getElementById("vulnerabilityCityFilter");
const wardReferenceRows = document.getElementById("wardReferenceRows");
const wardReferenceMessage = document.getElementById("wardReferenceMessage");
const wardReferenceTableWrap = document.getElementById("wardReferenceTableWrap");
const selectedWardCapacity = document.getElementById("selectedWardCapacity");
let operationalWards = [];

async function operationalRequest(url, options = {}) {
    const response = await fetch(url, {
        credentials: "same-origin",
        ...options,
        headers: {
            Accept: "application/json",
            ...(options.body ? { "Content-Type": "application/json" } : {}),
            ...(options.method && options.method !== "GET" ? { "X-CSRF-Token": csrfToken } : {}),
            ...(options.headers || {}),
        },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status}).`);
    return body;
}

function displayOperationalValue(value, digits = 2) {
    if (value === null || value === undefined || value === "") return "Not provided";
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString("en", { maximumFractionDigits: digits }) : "Not provided";
}

function addCell(row, value, className = "") {
    const cell = document.createElement("td");
    if (className) cell.className = className;
    cell.textContent = value;
    row.append(cell);
    return cell;
}

function renderWardReference() {
    if (!wardReferenceRows) return;
    const city = vulnerabilityCityFilter.value;
    const riskScore = { LOW: 0, MODERATE: 1 / 3, HIGH: 2 / 3, CRITICAL: 1 };
    const priority = (ward) => {
        const vulnerability = Number(ward.vulnerability_index);
        const tier = ward.forecast_by_horizon?.["2"]?.risk_tier;
        if (!Number.isFinite(vulnerability)) return -1;
        return tier && riskScore[tier] !== undefined
            ? 0.5 * vulnerability + 0.5 * riskScore[tier]
            : vulnerability;
    };
    const rows = operationalWards
        .filter((ward) => !city || ward.city === city)
        .sort((a, b) => priority(b) - priority(a));
    wardReferenceRows.replaceChildren();
    rows.forEach((ward, index) => {
        const tr = document.createElement("tr");
        const choose = document.createElement("button");
        choose.type = "button";
        choose.textContent = String(index + 1);
        choose.setAttribute("aria-label", `Select ${ward.ward_id}`);
        choose.addEventListener("click", () => selectWardFromReference(ward));
        const rank = document.createElement("td");
        rank.append(choose);
        tr.append(rank);
        addCell(tr, ward.ward_id);
        addCell(tr, ward.city);
        addCell(tr, displayOperationalValue(ward.vulnerability_index));
        addCell(tr, displayOperationalValue(ward.mean_weekly_cases_2016_2023));
        addCell(tr, displayOperationalValue(ward.estimated_bed_capacity, 0));
        const actual = ward.actual_available_beds_status === "reported"
            ? displayOperationalValue(ward.actual_available_beds, 0)
            : ward.actual_available_beds_status === "partial" ? "Partial reports" : "Not provided";
        addCell(tr, actual);
        const riskCell = addCell(tr, ward.forecast_by_horizon?.["2"]?.risk_tier || "Unavailable");
        if (ward.forecast_by_horizon?.["2"]?.risk_tier) riskCell.className = `risk-tier ${ward.forecast_by_horizon["2"].risk_tier}`;
        tr.addEventListener("click", (event) => {
            if (event.target !== choose) selectWardFromReference(ward);
        });
        wardReferenceRows.append(tr);
    });
}

function selectWardFromReference(ward) {
    if (citySelect && wardSelect) {
        citySelect.value = ward.city;
        citySelect.dispatchEvent(new Event("change"));
        wardSelect.value = ward.ward_id;
        wardSelect.dispatchEvent(new Event("change"));
    }
    const forecast = ward.forecast_by_horizon || {};
    selectedWardCapacity.replaceChildren();
    const values = [
        ["Selected ward", `${ward.city} · ${ward.ward_id}`],
        ["Vulnerability index (provisional)", displayOperationalValue(ward.vulnerability_index)],
        ["Historical weekly mean (2016–2023)", displayOperationalValue(ward.mean_weekly_cases_2016_2023)],
        ["Estimated bed capacity", displayOperationalValue(ward.estimated_bed_capacity, 0)],
        ["Actual available beds", ward.actual_available_beds_status === "reported" ? displayOperationalValue(ward.actual_available_beds, 0) : "Not provided"],
        ["Saline stock", ward.saline_stock_available_status === "reported" ? displayOperationalValue(ward.saline_stock_available, 0) : "Not provided"],
        ["Diagnostic kits", ward.diagnostic_kits_stock_available_status === "reported" ? displayOperationalValue(ward.diagnostic_kits_stock_available, 0) : "Not provided"],
        ["Available response crews", displayOperationalValue(ward.crew_count_available, 0)],
        ["2-week model risk", forecast["2"]?.risk_tier || "Unavailable until a compatible 34-feature model is trained"],
    ];
    values.forEach(([label, value]) => {
        const box = document.createElement("div");
        const name = document.createElement("span");
        const content = document.createElement("strong");
        name.textContent = label;
        content.textContent = value;
        box.append(name, content);
        selectedWardCapacity.append(box);
    });
}

async function loadWardReference() {
    if (!wardReferenceMessage) return;
    try {
        const payload = await operationalRequest("/api/wards/operational-summary");
        operationalWards = Array.isArray(payload.wards) ? payload.wards : [];
        const cities = [...new Set(operationalWards.map((ward) => ward.city))].sort();
        cities.forEach((city) => vulnerabilityCityFilter.add(new Option(city, city)));
        wardReferenceMessage.hidden = operationalWards.length > 0;
        wardReferenceMessage.textContent = operationalWards.length ? "" : "Ward reference data is not available.";
        wardReferenceTableWrap.hidden = operationalWards.length === 0;
        renderWardReference();
    } catch (error) {
        wardReferenceMessage.textContent = error.message;
    }
}

vulnerabilityCityFilter?.addEventListener("change", renderWardReference);
loadWardReference();

const hospitalForm = document.getElementById("hospitalInventoryForm");
const hospitalMessage = document.getElementById("hospitalInventoryMessage");
const hospitalHistory = document.getElementById("hospitalInventoryHistory");
async function loadHospitalInventory() {
    if (!hospitalMessage) return;
    try {
        const data = await operationalRequest("/api/hospital/operational-data");
        document.getElementById("hospitalAssociationLabel").textContent = `${data.hospital_name} · assigned wards: ${data.wards.length ? data.wards.map((item) => `${item.city} ${item.ward_id}`).join(", ") : "not assigned yet"}`;
        for (const name of ["actual_available_beds", "saline_stock_available", "diagnostic_kits_stock_available"]) {
            hospitalForm.elements[name].value = data.inventory[name] ?? "";
        }
        hospitalForm.hidden = false;
        hospitalMessage.textContent = "Blank values are unknown. Submit only values reported by your hospital.";
        const historyData = await operationalRequest("/api/hospital/operational-history");
        hospitalHistory.replaceChildren();
        (historyData.events || []).forEach((event) => {
            const li = document.createElement("li");
            li.textContent = `${formatDate(event.timestamp)} · ${event.description}`;
            hospitalHistory.append(li);
        });
        if (!historyData.events?.length) hospitalHistory.textContent = "No inventory changes have been recorded.";
    } catch (error) {
        hospitalMessage.textContent = error.message;
        hospitalForm.hidden = true;
        hospitalHistory.textContent = "History is available after the hospital association is verified.";
    }
}

hospitalForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {};
    for (const name of ["actual_available_beds", "saline_stock_available", "diagnostic_kits_stock_available"]) {
        const value = hospitalForm.elements[name].value.trim();
        payload[name] = value === "" ? null : Number(value);
    }
    try {
        await operationalRequest("/api/hospital/operational-data", { method: "PUT", body: JSON.stringify(payload) });
        hospitalMessage.textContent = "Hospital inventory saved and audited.";
        await loadHospitalInventory();
        await loadWardReference();
    } catch (error) { hospitalMessage.textContent = error.message; }
});
loadHospitalInventory();

const wardOpsPanel = document.getElementById("ward-operations");
const wardCrewForm = document.getElementById("wardCrewForm");
const wardOpsMessage = document.getElementById("wardOperationsMessage");
async function loadOperationsScope() {
    if (!wardOpsPanel) return;
    try {
        const data = await operationalRequest("/api/operations/wards");
        const select = wardCrewForm.elements.ward_id;
        select.replaceChildren();
        (data.wards || []).forEach((ward) => select.add(new Option(`${ward.city} · ${ward.ward_id}`, ward.ward_id)));
        wardOpsPanel.hidden = !(data.wards || []).length;
        wardCrewForm.hidden = !(data.wards || []).length;
        wardOpsMessage.textContent = data.wards?.length ? "You can update crew availability only in your assigned scope." : "No assigned operations wards.";
        if (data.wards?.length) select.dispatchEvent(new Event("change"));
    } catch (error) {
        wardOpsPanel.hidden = true;
        wardOpsMessage.textContent = error.message;
    }
}
wardCrewForm?.elements.ward_id.addEventListener("change", () => {
    const ward = operationalWards.find((item) => item.ward_id === wardCrewForm.elements.ward_id.value);
    if (ward) wardCrewForm.elements.crew_count_available.value = ward.crew_count_available ?? "";
});
wardCrewForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const ward = wardCrewForm.elements.ward_id.value;
    try {
        await operationalRequest(`/api/operations/wards/${encodeURIComponent(ward)}/crew`, {
            method: "PUT", body: JSON.stringify({ crew_count_available: Number(wardCrewForm.elements.crew_count_available.value) }),
        });
        wardOpsMessage.textContent = "Crew availability saved and audited.";
        await loadWardReference();
        await loadOperationsScope();
    } catch (error) { wardOpsMessage.textContent = error.message; }
});
loadOperationsScope();

const allocationForm = document.getElementById("allocationForm");
const allocationMessage = document.getElementById("allocationMessage");
const allocationResults = document.getElementById("allocationResults");
allocationForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    allocationMessage.textContent = "Checking compatible forecasts and reported stock…";
    allocationResults.hidden = true;
    const form = new FormData(allocationForm);
    const asOf = String(form.get("asOf") || "");
    const payload = {
        horizonWeeks: Number(form.get("horizonWeeks")),
        asOf: asOf || null,
        stockAvailabilityPct: Number(form.get("stockAvailabilityPct")),
        bedCapacityAdjustmentPct: Number(form.get("bedCapacityAdjustmentPct")),
        hypotheticalCrewCount: Number(form.get("hypotheticalCrewCount")),
    };
    try {
        const data = await operationalRequest("/api/resources/allocation", { method: "POST", body: JSON.stringify(payload) });
        allocationResults.replaceChildren();
        (data.city_summary || []).forEach((city) => {
            const p = document.createElement("p");
            p.textContent = `${city.city}: estimated bed headroom ${displayOperationalValue(city.estimated_capacity_headroom)}; remaining actual bed shortage ${displayOperationalValue(city.remaining_actual_bed_shortage)}.`;
            allocationResults.append(p);
        });
        (data.proposed_transfers || []).forEach((move) => {
            const p = document.createElement("p");
            p.textContent = `${move.city}: propose ${displayOperationalValue(move.quantity)} ${move.resource} from ${move.from_ward} to ${move.to_ward}; ${move.decision_status}.`;
            allocationResults.append(p);
        });
        allocationResults.hidden = false;
        allocationMessage.textContent = "Planning scenario calculated. All transfers are proposals requiring confirmation.";
    } catch (error) { allocationMessage.textContent = error.message; }
});

const whatIfForm = document.getElementById("whatIfForm");
const whatIfMessage = document.getElementById("whatIfMessage");
const whatIfResults = document.getElementById("whatIfResults");
const whatIfCity = document.getElementById("whatIfCity");
const whatIfWards = document.getElementById("whatIfWards");
const effectivenessInput = whatIfForm?.elements.interventionEffectivenessPct;
effectivenessInput?.addEventListener("input", () => {
    document.getElementById("effectivenessValue").textContent = `${effectivenessInput.value}%`;
});
whatIfCity?.addEventListener("change", () => {
    const option = citySelect.querySelector(`option[value="${CSS.escape(whatIfCity.value)}"]`);
    let wards = [];
    try { wards = JSON.parse(option?.dataset.wards || "[]"); } catch (_) { wards = []; }
    whatIfWards.replaceChildren();
    wards.forEach((ward) => whatIfWards.add(new Option(ward, ward)));
});
whatIfForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    whatIfMessage.textContent = "Checking the shared 34-feature model artifacts…";
    whatIfResults.hidden = true;
    const payload = {
        city: whatIfCity.value,
        selectedWards: [...whatIfWards.selectedOptions].map((option) => option.value),
        forecastDate: whatIfForm.elements.forecastDate.value,
        hypotheticalCrewCount: Number(whatIfForm.elements.hypotheticalCrewCount.value),
        interventionEffectivenessPct: Number(effectivenessInput.value),
    };
    try {
        const data = await operationalRequest("/api/what-if", { method: "POST", body: JSON.stringify(payload) });
        whatIfResults.replaceChildren();
        const warning = document.createElement("p");
        warning.textContent = data.warning;
        whatIfResults.append(warning);
        (data.forecasts || []).forEach((row) => {
            const line = document.createElement("p");
            line.textContent = `${row.ward_id} · ${row.horizon_weeks}w: ${displayOperationalValue(row.baseline_expected_cases)} baseline → ${displayOperationalValue(row.scenario_expected_cases)} scenario cases; ${displayOperationalValue(row.scenario_saline_bags - row.baseline_saline_bags)} saline bag change.`;
            whatIfResults.append(line);
        });
        whatIfResults.hidden = false;
        whatIfMessage.textContent = "Scenario run completed; it does not alter historical inputs or saved forecasts.";
    } catch (error) { whatIfMessage.textContent = error.message; }
});
