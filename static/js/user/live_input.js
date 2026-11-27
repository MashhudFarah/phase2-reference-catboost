"use strict";

const form = document.getElementById("liveForm");
const city = document.getElementById("city");
const ward = document.getElementById("ward");
const observationDate = document.getElementById("observationDate");
const dateHelp = document.getElementById("dateHelp");
const predictButton = document.getElementById("predictButton");
const formMessage = document.getElementById("formMessage");
const result = document.getElementById("result");
const riskBadge = document.getElementById("riskBadge");
const alertBox = document.getElementById("alertBox");
const previousContext = document.getElementById("previousContext");
const csrfToken = document.querySelector('meta[name="csrf-token"]').content;

const areas = JSON.parse(document.body.dataset.areas || "{}");

function moneylessNumber(value, digits = 2) {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function formatDate(value) {
    return new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" }).format(new Date(`${value}T00:00:00`));
}

function setMessage(text, error = false) {
    formMessage.textContent = text;
    formMessage.className = `form-message${error ? " error" : ""}`;
}

function populateWards() {
    ward.innerHTML = '<option value="">Select ward</option>';
    (areas[city.value] || []).forEach((wardId) => {
        const option = document.createElement("option");
        option.value = wardId;
        option.textContent = wardId;
        ward.appendChild(option);
    });
    ward.disabled = !city.value;
    observationDate.value = "";
    previousContext.hidden = true;
}

async function loadDefaults() {
    if (!city.value || !ward.value) return;
    setMessage("Loading ward context…");
    try {
        const response = await fetch(`/api/live-input/defaults?city=${encodeURIComponent(city.value)}&ward=${encodeURIComponent(ward.value)}`, {
            headers: { "Accept": "application/json" },
            credentials: "same-origin",
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not load ward defaults.");

        observationDate.value = data.nextExpectedWeeklyDate;
        observationDate.min = data.nextExpectedWeeklyDate;
        dateHelp.textContent = `Latest stored observation: ${formatDate(data.latestHistoricalDate)}. The live row must use ${formatDate(data.nextExpectedWeeklyDate)}.`;
        previousContext.hidden = false;
        previousContext.innerHTML = `Previous week: <strong>${moneylessNumber(data.previousCases, 0)} cases</strong> · ${moneylessNumber(data.previousRainfall)} mm rain · ${moneylessNumber(data.previousTemperature)} °C average temperature · ${moneylessNumber(data.previousHumidity)}% humidity`;
        setMessage("Ward context loaded. Enter this week's observations.");
    } catch (error) {
        setMessage(error.message, true);
    }
}

city.addEventListener("change", populateWards);
ward.addEventListener("change", loadDefaults);

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.hidden = true;
    predictButton.disabled = true;
    predictButton.textContent = "Running CatBoost…";
    setMessage("Building live features and running the trained model…");

    const payload = {
        city: city.value,
        ward: ward.value,
        observationDate: observationDate.value,
        horizonWeeks: Number(document.getElementById("horizon").value),
        avgTempC: Number(document.getElementById("avgTemp").value),
        maxTempC: Number(document.getElementById("maxTemp").value),
        minTempC: Number(document.getElementById("minTemp").value),
        rainfallMm: Number(document.getElementById("rainfall").value),
        humidityPct: Number(document.getElementById("humidity").value),
        dengueCases: Number(document.getElementById("dengueCases").value),
    };

    try {
        const response = await fetch("/api/live-predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": csrfToken,
                "Accept": "application/json",
            },
            credentials: "same-origin",
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Live prediction failed.");

        document.getElementById("predictedCases").textContent = moneylessNumber(data.predictedCases, 0);
        document.getElementById("targetDate").textContent = formatDate(data.targetDate);
        document.getElementById("targetHorizon").textContent = `${data.horizonWeeks} weeks`;
        document.getElementById("modelName").textContent = `${data.model} · ${data.modelTreeCount} trees`;
        riskBadge.textContent = data.riskCategory;
        riskBadge.dataset.risk = data.riskCategory;
        alertBox.textContent = data.warningMessage;
        alertBox.dataset.alert = String(data.earlyWarning);

        const inputSummary = document.getElementById("inputSummary");
        const inputs = [
            ["Observation date", formatDate(data.observationDate)],
            ["Average temperature", `${moneylessNumber(data.liveInput.avgTempC)} °C`],
            ["Maximum temperature", `${moneylessNumber(data.liveInput.maxTempC)} °C`],
            ["Minimum temperature", `${moneylessNumber(data.liveInput.minTempC)} °C`],
            ["Rainfall", `${moneylessNumber(data.liveInput.rainfallMm)} mm`],
            ["Humidity", `${moneylessNumber(data.liveInput.humidityPct)}%`],
            ["Current dengue cases", moneylessNumber(data.liveInput.dengueCases, 0)],
        ];
        inputSummary.innerHTML = inputs.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");

        document.getElementById("featureCount").textContent = data.modelInputFeatureCount;
        document.getElementById("artifactInfo").textContent = `Run: ${data.experimentId} · training data through ${formatDate(data.modelTrainingDataThrough)}.`;
        result.hidden = false;
        result.scrollIntoView({ behavior: "smooth", block: "start" });
        setMessage("Live early warning generated successfully.");
    } catch (error) {
        setMessage(error.message, true);
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Run live early warning →";
    }
});
