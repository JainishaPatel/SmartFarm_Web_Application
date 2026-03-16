// Inputs
  const cropInput = document.getElementById('crop_name');
  const quantityInput = document.getElementById('quantity');
  const unitInput = document.getElementById('unit');
  const priceInput = document.getElementById('price');
  const locationInput = document.getElementById('location');
  const sellerInput = document.getElementById('seller_name');
  const contactInput = document.getElementById('seller_contact');
  const descriptionInput = document.getElementById('description');
  const imageInput = document.getElementById('image');

  // Preview elements
  const previewTitle = document.querySelector('.listing-card-title');
  const previewQty = document.querySelector('.listing-card-quantity');
  const previewPrice = document.querySelector('.listing-card-price');
  const previewLocation = document.querySelector('.listing-card-location');
  const previewSeller = document.querySelector('.listing-card-seller');
  const previewContact = document.querySelector('.listing-card-contact');
  const previewDesc = document.querySelector('.listing-card-description');
  const previewImg = document.getElementById('preview-img');

  // Event listeners for editable fields
cropInput.addEventListener('input', () => previewTitle.textContent = cropInput.value || 'Crop Name');

function updateQuantity() {
  previewQty.textContent = 'Qty: ' + (quantityInput.value || '0') + ' ' + unitInput.value;
}
quantityInput.addEventListener('input', updateQuantity);
unitInput.addEventListener('change', updateQuantity);

function updatePrice() {
  const unit = unitInput.value || 'kg';
  const price = priceInput.value || '0';
  previewPrice.textContent = `Price: ₹${price} / ${unit}`;
}

// Listen for changes
priceInput.addEventListener('input', updatePrice);
unitInput.addEventListener('change', updatePrice);

locationInput.addEventListener('input', () => previewLocation.textContent = locationInput.value || 'Location');
descriptionInput.addEventListener('input', () => previewDesc.textContent = descriptionInput.value || 'Description');

// Initialize preview with pre-filled readonly values
previewSeller.textContent = sellerInput.value || 'Seller Name';
previewContact.textContent = contactInput.value || 'Contact Info';

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if(file){
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewImg.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    previewImg.src = '';
    previewImg.style.display = 'none';
  }
});
