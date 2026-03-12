function runDetection() {
    console.log("runDetection function started");
    
    const output = document.getElementById("output");
    const fileInput = document.getElementById("imageFile");
    
    console.log("Output element:", output);
    console.log("File input element:", fileInput);

    if (!output) {
        console.error("Output element not found!");
        return;
    }

    output.innerHTML = "<span style='color:blue'>⏳ Processing image...</span>";

    if (!fileInput || fileInput.files.length === 0) {
        output.innerHTML = "<span style='color:red'>❌ Please select an image</span>";
        return;
    }

    console.log("File selected:", fileInput.files[0].name);
    
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    console.log("Sending request to backend...");
    
    fetch("https://braincare1.onrender.com/detect", {
        method: "POST",
        body: formData
    })
    .then(res => {
        console.log("Response received, status:", res.status);
        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        console.log("Response data:", data);
        
        if (!data.success) {
            output.innerHTML = `<span style='color:red'>❌ Detection failed: ${data.message || 'Unknown error'}</span>`;
            return;
        }

        if (data.result === "Stroke Detected") {
            output.innerHTML = `
              <b style="color:red">STROKE DETECTED</b><br>
              Processing Time: ${data.processing_ms} ms
            `;
        } else {
            output.innerHTML = `
              <b style="color:green">NORMAL BRAIN</b><br>
              Processing Time: ${data.processing_ms} ms
            `;
        }
    })
    .catch(err => {
        console.error("Fetch error details:", err);
        output.innerHTML = "<span style='color:red'>❌ Cannot connect to server. Make sure backend is running on http://127.0.0.1:5000</span>";
    });
}