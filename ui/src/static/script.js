// Import speech utilities
import { initializeSpeechUtils, playTTSStream } from './speech_utils.js';

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const messageForm = document.getElementById('message-form');
const userInput = document.getElementById('user-input');
const metadataElement = document.getElementById('metadata');
const toggleSettingsButton = document.getElementById('toggle-settings');
const settingsPanel = document.getElementById('settings');
const clearChatButton = document.getElementById('clear-chat');
const settingsPopup = document.getElementById('settings-popup');
const closeSettingsButton = document.getElementById('close-settings');

// Settings Elements
const temperatureInput = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const maxTokensInput = document.getElementById('max-tokens');
const maxTokensValue = document.getElementById('max-tokens-value');
const topKInput = document.getElementById('top-k');
const topKValue = document.getElementById('top-k-value');
const bestOfInput = document.getElementById('best-of');
const bestOfValue = document.getElementById('best-of-value');
const repetitionPenaltyInput = document.getElementById('repetition-penalty');
const repetitionPenaltyValue = document.getElementById('repetition-penalty-value');

// New Database Select Element
const dbSelect = document.getElementById('database-select');

// Language buttons
const langButtons = document.querySelectorAll('.lang-button');

// Translations object
const translations = {
    'kk': {
        'Параметрлер': 'Параметрлер',
        'Қазақша': 'Қазақша',
        'Орысша': 'Орысша',
        'Ағылшынша': 'Ағылшынша',
        'Чатты тазалау': 'Чатты тазалау',
        'Хабарламаңызды осында теріңіз...': 'Мәтініңізді теріңіз…',
        'Дыбыс жазу': 'Дыбыс жазу',
        'TTS ауыстыру': 'TTS ауыстыру',
        'Жіберу': 'Жіберу',
        'Жүйелік нұсқау:': 'Жүйелік нұсқау:',
        'Сіз өте қабілетті AI көмекшісісіз. Жауаптар мысалдарсыз қысқа және қысқа болуы керек.': 'You are a highly knowledgeable assistant who should use language of the user query. Please answer in 3 sentences at most.',
        'Температура:': 'Температура:',
        'Макс токендер:': 'Макс токендер:',
        'Жоғарғы K:': 'Жоғарғы K:',
        'Ең жақсысы:': 'Ең жақсысы:',
        'Қайталау айыппұлы:': 'Қайталау айыппұлы:',
        'Рөлі:': 'Рөлі:',
        'Helpful AI Assistant': 'Helpful AI Assistant',
        'Ask a question about the image or press send to describe it': 'Сурет туралы сұрақ қойыңыз немесе сипаттау үшін жіберу түймесін басыңыз'
    },
    'en': {
        'Параметрлер': 'Settings',
        'Қазақша': 'Kazakh',
        'Орысша': 'Russian',
        'Ағылшынша': 'English',
        'Чатты тазалау': 'Clear chat',
        'Хабарламаңызды осында теріңіз...': 'Type your message here...',
        'Дыбыс жазу': 'Record audio',
        'TTS ауыстыру': 'Toggle TTS',
        'Жіберу': 'Send',
        'Жүйелік нұсқау:': 'System prompt:',
        'Сіз өте қабілетті AI көмекшісісіз. Жауаптар мысалдарсыз қысқа және қысқа болуы керек.': 'You are a highly knowledgeable assistant who should use language of the user query. Please answer in 3 sentences at most.',
        'Температура:': 'Temperature:',
        'Макс токендер:': 'Max tokens:',
        'Жоғарғы K:': 'Top K:',
        'Ең жақсысы:': 'Best of:',
        'Қайталау айыппұлы:': 'Repetition penalty:',
        'Рөлі:': 'Role:',
        'Helpful AI Assistant': 'Helpful AI Assistant',
        'Ask a question about the image or press send to describe it': 'Ask a question about the image or press send to describe it'
    },
    'ru': {
        'Параметрлер': 'Настройки',
        'Қазақша': 'Казахский',
        'Орысша': 'Русский',
        'Ағылшынша': 'Английский',
        'Чатты тазалау': 'Очистить чат',
        'Хабарламаңызды осында теріңіз...': 'Введите ваше сообщение здесь...',
        'Дыбыс жазу': 'Запись аудио',
        'TTS ауыстыру': 'Переключить TTS',
        'Жіберу': 'Отправить',
        'Жүйелік нұсқау:': 'Системная инструкция:',
        'Сіз өте қабілетті AI көмекшісісіз. Жауаптар мысалдарсыз қысқа және қысқа болуы керек.': 'You are a highly knowledgeable assistant who should use language of the user query. Please answer in 3 sentences at most.',
        'Температура:': 'Температура:',
        'Макс токендер:': 'Макс. токенов:',
        'Жоғарғы K:': 'Топ K:',
        'Ең жақсысы:': 'Лучший из:',
        'Қайталау айыппұлы:': 'Штраф за повторение:',
        'Рөлі:': 'Роль:',
        'Helpful AI Assistant': 'Helpful AI Assistant ru',
        'Ask a question about the image or press send to describe it': 'Задайте вопрос о изображении или нажмите отправить, чтобы описать его'
    }
};

let currentLanguage = 'kk';

// API Endpoints
const GENERATE_API_URL = '/generate';
const PREDICT_API_URL = '/predict';

let chatHistory = [];;

// Event Listeners
toggleSettingsButton.addEventListener('click', () => {
    settingsPanel.classList.toggle('active');
    toggleSettingsButton.classList.toggle('active');
});

clearChatButton.addEventListener('click', () => {
    chatContainer.innerHTML = '';
    metadataElement.textContent = '';
    chatHistory = [];
});

function updateSettingValue(input, valueElement) {
    valueElement.textContent = input.value;
    input.addEventListener('input', () => {
        valueElement.textContent = input.value;
    });
}

// Initialize setting values
updateSettingValue(temperatureInput, temperatureValue);
updateSettingValue(maxTokensInput, maxTokensValue);
updateSettingValue(topKInput, topKValue);
updateSettingValue(bestOfInput, bestOfValue);
updateSettingValue(repetitionPenaltyInput, repetitionPenaltyValue);

// Form submission
messageForm.addEventListener('submit', (e) => handleSubmit(e));

// Handle Enter key for submission
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
    }
});

// Get the system prompt element
const systemPromptInput = document.getElementById('system-prompt');


// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    const message = userInput.value.trim();
    
    if (!message) return;

    addMessage('user', message);
    const dbName = dbSelect.value;
    if (dbName) {
        await handleRAGRequest(message, dbName);
    } else {
        await handleChatRequest(message);
    }

    userInput.value = '';
    autoResizeTextarea(userInput);
    scrollToBottom();
}

// Handle chat request
async function handleChatRequest(message) {
    const startTime = Date.now();
    let aiMessage = '';
    let aiMessageElement = null;
    let buffer = '';
    let usageData = null;

    try {
        const requestBody = {
            messages: [...chatHistory, { role: "user", content: message }],
            temperature: parseFloat(temperatureInput.value),
            max_tokens: parseInt(maxTokensInput.value),
            top_k: parseInt(topKInput.value),
            best_of: parseInt(bestOfInput.value),
            repetition_penalty: parseFloat(repetitionPenaltyInput.value),
            system_prompt: systemPromptInput.value,
        };

        const response = await fetch(GENERATE_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            buffer += chunk;

            let lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.trim() === '') continue;
                if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (dataStr === '[DONE]') break;
                    try {
                        const data = JSON.parse(dataStr);

                        if (data.error) {
                            throw new Error(data.error);
                        }

                        if (data.usage) {
                            usageData = data.usage;
                            continue;
                        }

                        const content = data.choices[0].delta?.content || data.choices[0].text || '';

                        if (!aiMessageElement) {
                            aiMessageElement = addMessage('ai', '');
                        }

                        aiMessage += content;
                        const cleanHTML = DOMPurify.sanitize(marked.parse(aiMessage));
                        aiMessageElement.innerHTML = cleanHTML;
                    } catch (jsonError) {
                        console.error('Error parsing JSON:', jsonError);
                        console.error('Problematic data string:', dataStr);
                    }
                }
            }
        }

        chatHistory.push({ role: "user", content: message });
        chatHistory.push({ role: "assistant", content: aiMessage });

        const endTime = Date.now();
        const generationTime = (endTime - startTime) / 1000;

        let metadataText = `Generation time: ${generationTime.toFixed(2)} seconds`;
        if (usageData) {
            metadataText += ` | Input Tokens: ${usageData.prompt_tokens} | Output Tokens: ${usageData.completion_tokens} | Total Tokens: ${usageData.total_tokens}`;
        }
        metadataElement.textContent = metadataText;

        // Add copy button
        const copyButton = document.createElement('button');
        copyButton.classList.add('copy-button');
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.addEventListener('click', () => copyToClipboard(aiMessage, copyButton));
        aiMessageElement.appendChild(copyButton);

        // Play TTS if enabled
        playTTSStream(aiMessage);

    } catch (error) {
        console.error('Error during fetch:', error);
        addMessage('ai', `Please ask a valid question.`);
    }
}

// Handle RAG request
async function handleRAGRequest(message, dbName) {
    const startTime = Date.now();
    let aiMessage = '';
    let aiMessageElement = null;

    try {
        const requestBody = {
            message: message,
            system_prompt: systemPromptInput.value,
            temperature: parseFloat(temperatureInput.value),
            max_tokens: parseInt(maxTokensInput.value),
            top_k: parseInt(topKInput.value),
            db_name: dbName
        };

        const response = await fetch(PREDICT_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.response || `HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        console.log("RAG response:", responseData);

        aiMessage = responseData.response || '';
        aiMessageElement = addMessage('ai', aiMessage);

        if (responseData.retrieved_documents && responseData.retrieved_documents.length > 0) {
            addCitations(aiMessageElement, responseData.retrieved_documents);
        }

        chatHistory.push({ role: "user", content: message });
        chatHistory.push({ role: "assistant", content: aiMessage });

        const endTime = Date.now();
        const generationTime = (endTime - startTime) / 1000;

        let metadataText = `Generation time: ${generationTime.toFixed(2)} seconds | DB: ${dbName}`;
        metadataElement.textContent = metadataText;

        // Add copy button
        const copyButton = document.createElement('button');
        copyButton.classList.add('copy-button');
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.addEventListener('click', () => copyToClipboard(aiMessage, copyButton));
        aiMessageElement.appendChild(copyButton);

        // Play TTS if enabled
        playTTSStream(aiMessage);

    } catch (error) {
        console.error('Error during fetch:', error);
        addMessage('ai', `Please ask a valid question.`);
    }
}

// Settings popup handlers
toggleSettingsButton.addEventListener('click', () => {
    settingsPopup.style.display = 'block';
});

closeSettingsButton.addEventListener('click', () => {
    settingsPopup.style.display = 'none';
});

// Close settings popup when clicking outside
window.addEventListener('click', (event) => {
    if (event.target === settingsPopup) {
        settingsPopup.style.display = 'none';
    }
});

// Handle transcribed text
function handleTranscribedText(text) {
    const event = new Event('submit');
    userInput.value = text;
    handleSubmit(event);
}

// Add message to chat
function addMessage(sender, content) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender === 'user' ? 'user-message' : 'ai-message');

    const iconElement = document.createElement('i');
    iconElement.classList.add('fas', sender === 'user' ? 'fa-user' : 'fa-robot', 'message-icon');

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');

    if (typeof content === 'string') {
        contentElement.textContent = content;
    } else if (content instanceof HTMLElement) {
        contentElement.appendChild(content);
    }

    if (sender === 'user') {
        messageElement.appendChild(contentElement);
        messageElement.appendChild(iconElement);
    } else {
        messageElement.appendChild(iconElement);
        messageElement.appendChild(contentElement);
    }

    chatContainer.appendChild(messageElement);
    scrollToBottom();
    return contentElement;
}

// Scroll chat to bottom
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Translate interface
function translateInterface(lang) {
    currentLanguage = lang;

    // Update all elements with data-translate attribute
    document.querySelectorAll('[data-translate]').forEach(element => {
        const key = element.getAttribute('data-translate');
        if (translations[lang][key]) {
            if (element.tagName === 'INPUT' && element.type === 'text') {
                element.placeholder = translations[lang][key];
            } else if (element.tagName === 'OPTION') {
                element.text = translations[lang][key];
            } else {
                element.textContent = translations[lang][key];
            }
        }
    });
    
    // Update placeholder for textarea
    userInput.placeholder = translations[lang]['Хабарламаңызды осында теріңіз...'];
    
    // Update text in system prompt
    systemPromptInput.value = translations[lang]['Сіз өте қабілетті AI көмекшісісіз. Жауаптар мысалдарсыз қысқа және қысқа болуы керек.'];
    
    // Update labels for parameters
    document.querySelectorAll('.setting label').forEach(label => {
        const key = label.textContent.trim().replace(':', '');
        if (translations[lang][key]) {
            label.textContent = translations[lang][key] + ':';
        }
    });

    // Update options in select elements
    document.querySelectorAll('select option').forEach(option => {
        const key = option.textContent.trim();
        if (translations[lang][key]) {
            option.textContent = translations[lang][key];
        }
    });

    // Update button titles (tooltips)
    document.getElementById('clear-chat').title = translations[lang]['Чатты тазалау'];
    document.getElementById('record-button').title = translations[lang]['Дыбыс жазу'];
    document.getElementById('toggle-tts').title = translations[lang]['TTS ауыстыру'];
    document.getElementById('send-button').title = translations[lang]['Жіберу'];

    // Update image upload button text
    document.getElementById('upload-image').title = translations[lang]['Сурет жүктеу'];

    // Update settings popup title
    document.querySelector('#settings-popup h2').textContent = translations[lang]['Параметрлер'];

    // Update the placeholder text for image questions
    const imageQuestionPlaceholder = document.getElementById('image-question-placeholder');
    if (imageQuestionPlaceholder) {
        imageQuestionPlaceholder.textContent = translations[lang]['Ask a question about the image or press send to describe it'];
    }

    // Update the lang attribute of the html tag
    document.documentElement.lang = lang;

    // Highlight the active language button
    langButtons.forEach(button => {
        if (button.getAttribute('data-lang') === lang) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });

    console.log(`Interface language changed to: ${lang}`);
}

// Set language on server
async function setLanguageOnServer(lang) {
    try {
        const response = await fetch('/set_language', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ language: lang }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        console.log('Language set on server:', result);
        currentLanguage = lang; // Update the current language
    } catch (error) {
        console.error('Error setting language on server:', error);
    }
}

// Add citations to AI message
function addCitations(aiMessageElement, documents) {
    const citationsContainer = document.createElement('div');
    citationsContainer.classList.add('citations-container');

    documents.forEach((doc, index) => {
        const citationButton = document.createElement('button');
        citationButton.classList.add('citation-button');
        citationButton.textContent = `[${index + 1}]`;
        citationButton.addEventListener('click', () => {
            showDocumentContent(doc.content, doc.metadata, index + 1);
        });
        citationsContainer.appendChild(citationButton);
    });

    aiMessageElement.appendChild(citationsContainer);
}

// Show document content in modal
function showDocumentContent(content, metadata, citationNumber) {
    const modalOverlay = document.createElement('div');
    modalOverlay.classList.add('modal-overlay');

    const modalContent = document.createElement('div');
    modalContent.classList.add('modal-content');

    const modalHeader = document.createElement('div');
    modalHeader.classList.add('modal-header');
    modalHeader.textContent = `Document [${citationNumber}]`;

    const modalBody = document.createElement('div');
    modalBody.classList.add('modal-body');
    
    const contentParagraph = document.createElement('p');
    contentParagraph.textContent = content;
    modalBody.appendChild(contentParagraph);

    if (metadata) {
        const metadataHeader = document.createElement('h4');
        metadataHeader.textContent = 'Metadata:';
        modalBody.appendChild(metadataHeader);

        const metadataList = document.createElement('ul');
        for (const [key, value] of Object.entries(metadata)) {
            const listItem = document.createElement('li');
            listItem.textContent = `${key}: ${value}`;
            metadataList.appendChild(listItem);
        }
        modalBody.appendChild(metadataList);
    }

    const closeButton = document.createElement('button');
    closeButton.classList.add('modal-close-button');
    closeButton.textContent = 'Close';
    closeButton.addEventListener('click', () => {
        document.body.removeChild(modalOverlay);
    });

    modalContent.appendChild(modalHeader);
    modalContent.appendChild(modalBody);
    modalContent.appendChild(closeButton);
    modalOverlay.appendChild(modalContent);
    document.body.appendChild(modalOverlay);
}

// Copy to clipboard function
function copyToClipboard(text, button) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    
    textArea.select();
    
    try {
        document.execCommand('copy');
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.classList.add('copied');
        setTimeout(() => {
            button.innerHTML = originalIcon;
            button.classList.remove('copied');
        }, 2000);
    } catch (err) {
        console.error('Error in copying text: ', err);
    }
    
    document.body.removeChild(textArea);
}

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}

// Add input event listener for auto-resizing textarea
userInput.addEventListener('input', function() {
    autoResizeTextarea(this);
});

// Language button event listeners
langButtons.forEach(button => {
    button.addEventListener('click', async () => {
        const newLang = button.getAttribute('data-lang');
        await setLanguageOnServer(newLang);
        translateInterface(newLang);
    });
});

// Initialize speech utilities
initializeSpeechUtils('record-button', 'toggle-tts', handleTranscribedText);

// Set initial language on page load
document.addEventListener('DOMContentLoaded', () => {
    translateInterface('kk');
});

// Export necessary functions
export { handleTranscribedText };