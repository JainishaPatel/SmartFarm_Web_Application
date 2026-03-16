// crop_shelter.js

document.addEventListener("DOMContentLoaded", () => {
  // Animate cards and images on scroll
  const observerTargets = document.querySelectorAll(".feature-card, .uniform-img");

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("fade-in-up");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  observerTargets.forEach(target => observer.observe(target));

  // Click to enlarge gallery image (optional)
  const images = document.querySelectorAll(".uniform-img");
  images.forEach(img => {
    img.addEventListener("click", () => {
      const src = img.src;
      const modal = document.createElement("div");
      modal.classList.add("fullscreen-modal");
      modal.innerHTML = `
        <div class="fullscreen-backdrop" onclick="this.parentElement.remove()"></div>
        <img src="${src}" class="fullscreen-image" />
      `;
      document.body.appendChild(modal);
    });
  });
});