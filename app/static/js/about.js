document.addEventListener("DOMContentLoaded", () => {
  const elements = document.querySelectorAll(".fade-in");

  const showOnScroll = () => {
    elements.forEach(el => {
      const top = el.getBoundingClientRect().top;
      if (top < window.innerHeight * 0.85) {
        el.classList.add("visible");
      }
    });
  };

  window.addEventListener("scroll", showOnScroll);
  showOnScroll();
});