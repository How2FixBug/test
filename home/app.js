(() => {
  const form = document.getElementById("command-form");
  const input = document.getElementById("command-input");
  const output = document.getElementById("output");

  form.addEventListener("submit", (event) => {
    event.preventDefault();

    const value = input.value.trim();
    if (!value) {
      input.focus();
      return;
    }

    const entered = document.createElement("p");
    entered.className = "hint";
    entered.textContent = `guest@ultragpt:~$ ${value}`;

    const error = document.createElement("p");
    error.className = "error";
    error.textContent = "ERROR: NO RESPONSE FOUND";

    output.append(entered, error);
    input.value = "";
    input.focus();
  });
})();