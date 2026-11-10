const ChatHistory = (() => {
    const storageKey = "atomic-habits-chats-v1";

    function isValidMessage(message) {
        return message &&
            (message.role === "user" || message.role === "assistant") &&
            typeof message.text === "string" &&
            (message.sources === undefined || Array.isArray(message.sources)) &&
            (message.error === undefined || typeof message.error === "boolean");
    }

    function load(storage) {
        const raw = storage.getItem(storageKey);
        if (!raw) return [];

        const chats = JSON.parse(raw);
        if (!Array.isArray(chats) || !chats.every(chat =>
            chat && typeof chat.id === "string" &&
            typeof chat.title === "string" &&
            Array.isArray(chat.messages) && chat.messages.every(isValidMessage)
        )) {
            throw new Error("Saved chat history is invalid.");
        }
        return chats;
    }

    function save(storage, chats) {
        storage.setItem(storageKey, JSON.stringify(chats));
    }

    function create(id, question) {
        return { id, title: question.slice(0, 60), messages: [] };
    }

    return { load, save, create };
})();

if (typeof module !== "undefined") module.exports = ChatHistory;
