document.addEventListener('DOMContentLoaded', function () {
  // --- Hero section text animations ---
  document.querySelectorAll(".animate-fade-slide").forEach(el => {
    el.classList.add("fade-slide-in");
  });

  document.querySelectorAll(".animate-fade-delay").forEach(el => {
    el.classList.add("fade-delay-in");
  });

  document.querySelectorAll(".animate-pop-in").forEach(el => {
    el.classList.add("pop-in");
  });

  // --- Feature card scroll animation ---
  const featureCards = document.querySelectorAll(".feature-card");

  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("fade-in-up");
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
  });

  featureCards.forEach(card => observer.observe(card));
});

