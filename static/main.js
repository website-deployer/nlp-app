document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('signup-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const email = document.getElementById('email').value.trim();
            if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
                alert('Please enter a valid email address.');
                return;
            }
            form.innerHTML = '<div style="text-align:center;padding:1rem 0;font-size:1.2rem;color:#34C759;">Thank you for signing up!</div>';
        });
    }
}); 