const button = document.getElementById("test-button");
const result = document.getElementById("result");

button.addEventListener("click", async () => {
    try {
        const response = await fetch("http://127.0.0.1:8000/api/hello");

        const data = await response.json();

        result.textContent = data.message;
    } catch (error) {
        result.textContent = "Could not connect to backend.";
        console.error(error);
    }
});
