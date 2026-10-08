const status = document.getElementById("contact-status");
async function copy(value, message) {
  try {
    await navigator.clipboard.writeText(value);
    status.textContent = message;
  } catch {
    status.textContent = "Copy is unavailable here. The email address and URL are shown below.";
  }
}
document.getElementById("copy-email")?.addEventListener("click", (event) => copy(event.currentTarget.dataset.email, "Email address copied."));
document.getElementById("share-profile")?.addEventListener("click", async (event) => {
  const url = event.currentTarget.dataset.url;
  if (navigator.share) {
    try {
      await navigator.share({ title: "Hyeongyu Kim", text: "Research and contact", url });
    } catch (error) {
      if (error.name !== "AbortError") await copy(url, "Profile URL copied.");
    }
  } else await copy(url, "Profile URL copied.");
});
