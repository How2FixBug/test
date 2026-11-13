(() => {
  const form = document.getElementById("command-form");
  const input = document.getElementById("command-input");
  const output = document.getElementById("output");
  const cursor = document.querySelector(".cursor");

  const positionCursor = () => {
    const style = getComputedStyle(input);
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    context.font = style.font;
    const beforeCaret = input.value.slice(0, input.selectionStart ?? input.value.length);
    const textWidth = context.measureText(beforeCaret).width;
    cursor.style.left = `${textWidth - input.scrollLeft}px`;
  };

  input.addEventListener("input", positionCursor);
  input.addEventListener("keyup", positionCursor);
  input.addEventListener("click", positionCursor);
  input.addEventListener("select", positionCursor);
  input.addEventListener("scroll", positionCursor);
  input.addEventListener("focus", positionCursor);
  input.addEventListener("blur", () => { cursor.style.opacity = "0"; });
  input.addEventListener("focus", () => { cursor.style.opacity = ".8"; });

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
    positionCursor();
    input.focus();
  });
})();