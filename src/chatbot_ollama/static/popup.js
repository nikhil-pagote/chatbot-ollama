function toggleChat() {
    const popup = document.getElementById("chat-popup");
    popup.style.display = popup.style.display === "none" || popup.style.display === "" ? "block" : "none";
  }

  async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (!message) return;

    const messagesDiv = document.getElementById("messages");
    messagesDiv.innerHTML += `<p><strong>You:</strong> ${message}</p>`;
    input.value = "";

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      const data = await res.json();
      messagesDiv.innerHTML += `<p><strong>Bot:</strong> ${data.reply}</p>`;
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } catch (err) {
      messagesDiv.innerHTML += `<p><strong>Bot:</strong> Error reaching server.</p>`;
    }
  }
