document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("contactForm");
  const name = document.getElementById("name");
  const email = document.getElementById("email");
  const message = document.getElementById("message");

  form.addEventListener("submit", (e) => {
    // Clear previous errors
    document.getElementById("nameError").style.display = "none";
    document.getElementById("emailError").style.display = "none";
    document.getElementById("messageError").style.display = "none";

    let valid = true;

    // Name validation
    if (!name.value.trim()) {
      document.getElementById("nameError").style.display = "block";
      valid = false;
    }

    // Email validation (basic regex)
    const emailPattern = /^[^ ]+@[^ ]+\.[a-z]{2,3}$/;
    if (!email.value.trim() || !emailPattern.test(email.value.trim())) {
      document.getElementById("emailError").style.display = "block";
      valid = false;
    }

    // Message validation
    if (!message.value.trim()) {
      document.getElementById("messageError").style.display = "block";
      valid = false;
    }

    if (!valid) {
      e.preventDefault(); // Stop submission if validation fails
    }
    // If valid, form submits normally to Flask /contact route
  });
});