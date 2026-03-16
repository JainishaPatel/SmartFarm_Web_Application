document.addEventListener("DOMContentLoaded", function () {
  const loginForm = document.getElementById("loginForm");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const emailError = document.getElementById("emailError");
  const passwordError = document.getElementById("passwordError");
  const togglePassword = document.getElementById("togglePassword");
  const eyeIcon = togglePassword.querySelector("i");
  

  // Toggle password visibility
  togglePassword.addEventListener("click", function () {
    const type = passwordInput.type === "password" ? "text" : "password";
    passwordInput.type = type;
    eyeIcon.classList.toggle("fa-eye");
    eyeIcon.classList.toggle("fa-eye-slash");
  });

  // Real-time error removal on input
  emailInput.addEventListener("input", () => {
    emailError.style.display = "none";
    emailInput.classList.remove("is-invalid");
  });

  passwordInput.addEventListener("input", () => {
    passwordError.style.display = "none";
    passwordInput.classList.remove("is-invalid");
  });

  // Form validation
  loginForm.addEventListener("submit", function (e) {
    let isValid = true;

    // Email validation
    const emailValue = emailInput.value.trim();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailValue)) {
      emailError.style.display = "block";
      emailInput.classList.add("is-invalid");
      isValid = false;
    } else {
      emailError.style.display = "none";
      emailInput.classList.remove("is-invalid");
    }

    // Password validation
    const passwordValue = passwordInput.value.trim();
    if (passwordValue === "") {
      passwordError.style.display = "block";
      passwordInput.classList.add("is-invalid");
      isValid = false;
    } else {
      passwordError.style.display = "none";
      passwordInput.classList.remove("is-invalid");
    }

    // Prevent form submission if invalid
    if (!isValid) {
      e.preventDefault();
    }
  });


});