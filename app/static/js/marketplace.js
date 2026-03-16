document.addEventListener("DOMContentLoaded", () => {
  const listingsRoot = document.getElementById("listings-root");
  const searchInput = document.getElementById("search-input");

  // Fetch listings from API
  async function fetchListings(query = "") {
    listingsRoot.innerHTML = '<div class="text-center py-5 text-muted">Loading listings…</div>';
    try {
      const res = await fetch(`/api/listings?q=${encodeURIComponent(query)}`);
      const data = await res.json();

      if (data.error) {
        listingsRoot.innerHTML = `<p class="text-danger text-center py-5">${data.error}</p>`;
        return;
      }

      if (!data.length) {
        listingsRoot.innerHTML = `<p class="text-center py-5 text-muted">No listings found.</p>`;
        return;
      }

      listingsRoot.innerHTML = data.map(l => {
        const safeData = encodeURIComponent(JSON.stringify(l));
        const imgUrl = l.image_url || "/static/assets/default.png";

        return `
          <div class="col-lg-3 col-md-4 col-sm-6">
            <div class="card shadow-sm h-100 listing-card">
              <img src="${imgUrl}" class="card-img-top" alt="${l.crop_name || 'No Image'}">
              <div class="card-body">
                <h5 class="card-title">${l.crop_name}</h5>
                <p class="card-text">Qty: ${l.quantity} ${l.unit || 'kg'} | Price: ₹${l.price} / ${l.unit}</p>
                <p class="card-text"><small class="text-muted">${l.location}</small></p>
              </div>
              <div class="card-footer text-end">
                <button class="btn btn-outline-primary btn-sm" 
                        onclick="showDetails('${safeData}')">
                  View
                </button>
              </div>
            </div>
          </div>
        `;
      }).join('');

    } catch (err) {
      console.error(err);
      listingsRoot.innerHTML = `<p class="text-danger text-center py-5">Failed to load listings.</p>`;
    }
  }

  // Show details in modal
  window.showDetails = (listing) => {
    const data = typeof listing === "string" 
      ? JSON.parse(decodeURIComponent(listing)) 
      : listing;

    document.getElementById("modalTitle").textContent = data.crop_name;
    document.getElementById("modalBody").innerHTML = `
      <img src="${data.image_url || '/static/assets/default.png'}" 
           class="modal-img-small">
      <p><strong>Quantity:</strong> ${data.quantity} ${data.unit || 'kg'}</p>
      <p><strong>Price:</strong> ₹${data.price} / ${data.unit}</p>
      <p><strong>Location:</strong> ${data.location}</p>
      <p><strong>Seller:</strong> ${data.seller_name || '-'}</p>
      <p><strong>Contact:</strong> ${data.contact || data.seller_email || '-'}</p>
      <p><strong>Description:</strong> ${data.description || '-'}</p>
    `;
    new bootstrap.Modal(document.getElementById("listingModal")).show();
  };

  // Fetch all listings initially
  fetchListings();

  // Search functionality
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.trim();
    fetchListings(query);
  });
});