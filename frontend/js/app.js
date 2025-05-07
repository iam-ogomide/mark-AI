document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const githubRepoInput = document.getElementById('github-repo');
    const githubBranchInput = document.getElementById('github-branch');
    const processGithubBtn = document.getElementById('process-github');
    const documentUpload = document.getElementById('document-upload');
    const processDocBtn = document.getElementById('process-doc');
    const filePreview = document.getElementById('file-preview');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    const uploadArea = document.getElementById('upload-area');
    const themeToggle = document.getElementById('theme-toggle');

    // API Base URL
    const API_BASE_URL = 'http://localhost:8000';

    // Initialize chat
    displayWelcomeMessage();

    // Sidebar Toggle
    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
    });

    // Theme Toggle
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        const icon = themeToggle.querySelector('i');
        if (document.body.classList.contains('dark-mode')) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
        }
    });

    // Upload Area Interactions
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--accent-color)';
        uploadArea.style.backgroundColor = 'rgba(72, 149, 239, 0.1)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#ddd';
        uploadArea.style.backgroundColor = 'transparent';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ddd';
        uploadArea.style.backgroundColor = 'transparent';
        
        if (e.dataTransfer.files.length) {
            documentUpload.files = e.dataTransfer.files;
            handleFileUpload({ target: documentUpload });
        }
    });

    uploadArea.addEventListener('click', () => {
        documentUpload.click();
    });

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    processGithubBtn.addEventListener('click', processGithubRepo);
    documentUpload.addEventListener('change', handleFileUpload);
    processDocBtn.addEventListener('click', processDocument);

    // Functions
    function displayWelcomeMessage() {
        const welcomeSection = document.querySelector('.welcome-message');
        if (welcomeSection) welcomeSection.style.display = 'block';
    }

    function addMessageToChat(role, content, timestamp) {
        const welcomeSection = document.querySelector('.welcome-message');
        if (welcomeSection && welcomeSection.style.display !== 'none') {
            welcomeSection.style.display = 'none';
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        metaDiv.innerHTML = `<span>${role.charAt(0).toUpperCase() + role.slice(1)}</span><span>${timestamp}</span>`;
        
        messageDiv.appendChild(metaDiv);
        
        if (typeof content === 'string') {
            // Process markdown-like formatting
            const processedContent = renderMarkdownLikeText(content);
            messageDiv.innerHTML += processedContent;
        } else if (content && typeof content === 'object') {
            if (content.response && content.response.text) {
                const processedContent = renderMarkdownLikeText(content.response.text);
                messageDiv.innerHTML += processedContent;
                
                if (content.response.code_snippets) {
                    for (const [lang, snippet] of Object.entries(content.response.code_snippets)) {
                        const codeDiv = document.createElement('pre');
                        codeDiv.className = 'code-block';
                        const langSpan = document.createElement('span');
                        langSpan.className = 'code-language';
                        langSpan.textContent = lang;
                        codeDiv.appendChild(langSpan);
                        codeDiv.appendChild(document.createTextNode('\n' + snippet));
                        messageDiv.appendChild(codeDiv);
                    }
                }
            }
            else if (content.text) {
                const processedContent = renderMarkdownLikeText(content.text);
                messageDiv.innerHTML += processedContent;
            }
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function renderMarkdownLikeText(text) {
        if (!text) return '';
        
        // Convert markdown-like formatting to HTML
        let html = text
            // Convert **bold** to <strong>
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Convert *italic* to <em>
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Convert `code` to <code>
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Convert bullet points to list items
            .replace(/^-\s(.*)$/gm, '<li>$1</li>')
            // Convert numbered lists
            .replace(/^\d+\.\s(.*)$/gm, '<li>$1</li>')
            // Convert line breaks to <br>
            .replace(/\n/g, '<br>');
        
        // Wrap lists in <ul> tags
        html = html.replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');
        
        // Convert ```code blocks``` to <pre><code>
        html = html.replace(/```([^`]+)```/g, '<pre class="code-block"><code>$1</code></pre>');
        
        // Convert headings (##) to <h3>
        html = html.replace(/^##\s(.*)$/gm, '<h3>$1</h3>');
        
        // Convert paragraphs (double newlines)
        html = html.replace(/(<br>\s*){2,}/g, '</p><p>');
        html = '<p>' + html + '</p>';
        
        return html;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        const timestamp = new Date().toLocaleString();
        addMessageToChat('user', message, timestamp);
        userInput.value = '';

        sendButton.innerHTML = '<div class="spinner"></div>';
        sendButton.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: message,
                    role: 'user'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            addMessageToChat('assistant', data, new Date().toLocaleString());
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToChat('assistant', {
                text: 'Sorry, I encountered an error processing your request. Please try again.'
            }, new Date().toLocaleString());
        } finally {
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
            sendButton.disabled = false;
        }
    }

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        processDocBtn.disabled = false;
        filePreview.innerHTML = '';

        if (file.type === 'text/plain') {
            const reader = new FileReader();
            reader.onload = function(e) {
                filePreview.innerHTML = `<strong>File Preview:</strong> (First 500 characters)<br><br>${e.target.result.substring(0, 500)}...`;
            };
            reader.readAsText(file);
        } else if (file.type === 'application/pdf') {
            filePreview.innerHTML = '<strong>File Preview:</strong> PDF preview not available';
        } else {
            filePreview.innerHTML = '<strong>Unsupported file type</strong>';
            processDocBtn.disabled = true;
        }
    }

    async function processGithubRepo() {
        const repoUrl = githubRepoInput.value.trim();
        const branch = githubBranchInput.value.trim() || 'main';

        if (!repoUrl) {
            alert('Please enter a GitHub repository URL');
            return;
        }

        processGithubBtn.innerHTML = '<div class="spinner"></div> Processing...';
        processGithubBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/process-github`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: repoUrl,
                    branch: branch
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            alert(data.message || 'Repository processed successfully!');
        } catch (error) {
            console.error('Error processing GitHub repo:', error);
            alert(`Error: ${error.message}`);
        } finally {
            processGithubBtn.innerHTML = '<i class="fas fa-code-branch"></i> Process Repository';
            processGithubBtn.disabled = false;
        }
    }

    async function processDocument() {
        const file = documentUpload.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        processDocBtn.innerHTML = '<div class="spinner"></div> Processing...';
        processDocBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/api/process-document`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            alert(data.message || 'Document processed successfully!');
        } catch (error) {
            console.error('Error processing document:', error);
            alert(`Error: ${error.message}`);
        } finally {
            processDocBtn.innerHTML = '<i class="fas fa-upload"></i> Process Document';
            processDocBtn.disabled = false;
            documentUpload.value = '';
            filePreview.innerHTML = '';
        }
    }
});