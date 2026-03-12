function runPrediction() {
    const output = document.getElementById("predictionOutput");
    output.innerHTML = "⏳ Predicting risk... Please wait";

    // Collect inputs
    const data = {
        age: Number(document.getElementById("age").value),
        hypertension: Number(document.getElementById("hypertension").value),
        avg_glucose_level: Number(document.getElementById("glucose").value),
        bmi: Number(document.getElementById("bmi").value),
        smoking_status: Number(document.getElementById("smoking").value)
    };

    // Basic validation
    for (let key in data) {
        if (isNaN(data[key]) || data[key] === "") {
            output.innerHTML = "❌ Please fill all fields correctly";
            return;
        }
    }

    fetch("https://braincare1.onrender.com/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(res => {

        if (!res.success) {
            output.innerHTML = "❌ Server Error";
            return;
        }

        // Format probability to 2 decimal places
        const percent = res.risk_percentage.toFixed(2);

        // Color based on result
        let color = "green";
        if (res.result.includes("High")) color = "red";
        else if (res.result.includes("Moderate")) color = "orange";

        output.innerHTML = `
            <div style="margin-top:15px; font-size:18px; font-weight:bold; color:${color};">
                ${res.result}
            </div>
            <div style="margin-top:8px; font-size:16px;">
                Risk Probability: <b>${percent}%</b>
            </div>
        `;
    })
    .catch(error => {
        console.error(error);
        output.innerHTML = "❌ Network / Server Error";
    });
}