async function predictCrop() {
    if (!navigator.geolocation) {
        alert("Geolocation not supported. Please enter manually.");
        return;
    }

    navigator.geolocation.getCurrentPosition(async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;

        try {
            const response = await fetch("/predict_auto", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lat, lon })
            });

            const data = await response.json();

            if (data.error) {
                document.getElementById("prediction").innerText = "Error: " + data.error;
            } else {
                document.getElementById("prediction").innerText = `🌱 Recommended Crop: ${data.prediction}\n🌡 Temp: ${data.temperature}°C\n💧 Humidity: ${data.humidity}%`;
            }
        } catch (err) {
            document.getElementById("prediction").innerText = "Network error: " + err;
        }
    }, (err) => {
        alert("Geolocation denied or unavailable. Please enter manually.");
    });
}

// Call automatically on page load
window.addEventListener("DOMContentLoaded", predictCrop);