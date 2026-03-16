document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("signupForm");

  const nameInput = document.getElementById("name");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");

  const nameError = document.getElementById("nameError");
  const emailError = document.getElementById("emailError");
  const passwordError = document.getElementById("passwordError");

  const togglePassword = document.getElementById("togglePassword");
  const eyeIcon = togglePassword.querySelector("i");

  const googleBtn = document.getElementById("googleSignupBtn");

  // Toggle password visibility
  togglePassword.addEventListener("click", () => {
    const type = passwordInput.type === "password" ? "text" : "password";
    passwordInput.type = type;
    eyeIcon.classList.toggle("fa-eye");
    eyeIcon.classList.toggle("fa-eye-slash");
  });

  // Validation on form submit
  form.addEventListener("submit", function (e) {
    let valid = true;

    // Name
    if (nameInput.value.trim() === "") {
      nameError.style.display = "block";
      nameInput.classList.add("is-invalid");
      valid = false;
    } else {
      nameError.style.display = "none";
      nameInput.classList.remove("is-invalid");
    }

    // Email
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(emailInput.value.trim())) {
      emailError.style.display = "block";
      emailInput.classList.add("is-invalid");
      valid = false;
    } else {
      emailError.style.display = "none";
      emailInput.classList.remove("is-invalid");
    }

    // Password
    if (passwordInput.value.trim().length < 6) {
      passwordError.style.display = "block";
      passwordInput.classList.add("is-invalid");
      valid = false;
    } else {
      passwordError.style.display = "none";
      passwordInput.classList.remove("is-invalid");
    }

    if (!valid) {
      e.preventDefault(); // Stop form submission
    }
  });

  // Google Signup redirect
  googleBtn.addEventListener("click", function () {
    window.location.href = "{{ url_for('google.login') }}";
  });
  
});