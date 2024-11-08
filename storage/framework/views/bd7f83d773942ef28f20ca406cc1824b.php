<!DOCTYPE html>
<html lang="<?php echo e(str_replace('_', '-', app()->getLocale())); ?>">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title><?php echo e(config('app.name')); ?></title>
    <?php echo app('Illuminate\Foundation\Vite')(['resources/css/app.css', 'resources/js/app.js']); ?>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
</head>
<body class="bg-gray-100">
    <div id="app">
        <nav class="bg-white shadow-lg">
            <div class="max-w-7xl mx-auto px-4">
                <div class="flex justify-between h-16">
                    <div class="flex">
                        <div class="flex-shrink-0 flex items-center">
                            <h1 class="text-xl font-bold">Weather Prediction AI</h1>
                        </div>
                    </div>
                </div>
            </div>
        </nav>

        <div class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <div class="px-4 py-6 sm:px-0">
                <div class="flex gap-4">
                    <!-- Weather Prediction Section -->
                    <div class="flex-1 bg-white rounded-lg shadow-xl p-6 animate__animated animate__fadeIn">
                        <h2 class="text-lg font-semibold mb-4">City Weather Prediction</h2>
                        <div class="space-y-4">
                            <input type="text" id="city" class="w-full border rounded p-2" placeholder="Enter city name">
                            <select id="duration" class="w-full border rounded p-2">
                                <option value="30">1 Month</option>
                                <option value="180">6 Months</option>
                                <option value="365">1 Year</option>
                                <option value="730">2 Years</option>
                            </select>
                            <button onclick="predictWeather()" class="w-full bg-blue-500 text-white rounded p-2 hover:bg-blue-600">
                                Predict Weather
                            </button>
                        </div>
                        <div id="prediction-results" class="mt-4"></div>
                    </div>

                    <!-- Image Upload Section -->
                    <div class="flex-1 bg-white rounded-lg shadow-xl p-6 animate__animated animate__fadeIn">
                        <h2 class="text-lg font-semibold mb-4">Weather Image Analysis</h2>
                        <div class="space-y-4">
                            <input type="file" id="weather-image" class="w-full" accept="image/*">
                            <button onclick="predictImage()" class="w-full bg-green-500 text-white rounded p-2 hover:bg-green-600">
                                Analyze Image
                            </button>
                        </div>
                        <div id="image-results" class="mt-4"></div>
                    </div>
                </div>

                <!-- Chatbot Section -->
                <div class="mt-6 bg-white rounded-lg shadow-xl p-6 animate__animated animate__fadeIn">
                    <h2 class="text-lg font-semibold mb-4">Weather Assistant</h2>
                    <div id="chat-messages" class="h-64 overflow-y-auto border rounded p-4 mb-4"></div>
                    <div class="flex gap-2">
                        <input type="text" id="chat-input" class="flex-1 border rounded p-2" placeholder="Ask about weather...">
                        <button onclick="sendMessage()" class="bg-purple-500 text-white rounded px-4 py-2 hover:bg-purple-600">
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <?php echo app('Illuminate\Foundation\Vite')('resources/js/app.js'); ?>
    <script>
    function sendMessage() {
        const chatInput = document.getElementById('chat-input');
        const message = chatInput.value.trim();
        if (!message) return;

        // Display the user message
        const chatMessages = document.getElementById('chat-messages');
        const userMessageDiv = document.createElement('div');
        userMessageDiv.classList.add('user-message', 'mb-2', 'text-right', 'bg-blue-100', 'p-2', 'rounded');
        userMessageDiv.textContent = message;
        chatMessages.appendChild(userMessageDiv);

        // Send the message to the backend
        fetch('<?php echo e(route('chatbot.response')); ?>', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-TOKEN': '<?php echo e(csrf_token()); ?>'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = data.response;

            // Display the bot response
            const botMessageDiv = document.createElement('div');
            botMessageDiv.classList.add('bot-message', 'mb-2', 'text-left', 'bg-gray-100', 'p-2', 'rounded');
            botMessageDiv.textContent = botMessage;
            chatMessages.appendChild(botMessageDiv);

            // Scroll to the bottom of the chat
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });

        // Clear input
        chatInput.value = '';
    }
</script>

</body>
</html><?php /**PATH D:\Vult\tracker\resources\views/weather/index.blade.php ENDPATH**/ ?>