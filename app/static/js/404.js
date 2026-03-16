document.getElementById("searchBtn").addEventListener("click", function() {
    const query = document.getElementById("searchInput").value.trim();
    if(query) {
        // Redirect to marketplace search page
        const url = `/farmer_marketplace?q=${encodeURIComponent(query)}`;
        window.location.href = url;
    } else {
        alert("Please enter a search term!");
    }
});

// Optional: Enter key triggers search
document.getElementById("searchInput").addEventListener("keypress", function(e) {
    if(e.key === "Enter") {
        document.getElementById("searchBtn").click();
    }
});