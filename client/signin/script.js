import Cookies from "js-cookie";
const hashValue = (val) =>
  crypto.subtle
    .digest("SHA-256", new TextEncoder("utf-8").encode(val))
    .then((h) => {
      let hexes = [],
        view = new DataView(h);
      for (let i = 0; i < view.byteLength; i += 4)
        hexes.push(("00000000" + view.getUint32(i).toString(16)).slice(-8));
      return hexes.join("");
    });
const usernameInput = document.getElementById("username-input");
const passwordInput = document.getElementById("password-input");
document.getElementById("signin-button").addEventListener("click", async () => {
  const username = usernameInput.value;
  const password = passwordInput.value;
  const hash = await hashValue(`${username}|${password}`);
  const user = await (
    await fetch(
      "http://127.0.0.1:5000/user?" +
        new URLSearchParams({
          userhash: hash,
        })
    )
  ).json();
  if (user) {
    Cookies.set("userhash", hash);
    window.location.pathname = "/";
  }
});
