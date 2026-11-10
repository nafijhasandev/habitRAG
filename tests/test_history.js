const test = require("node:test");
const assert = require("node:assert/strict");
const ChatHistory = require("../web/static/history.js");

function memoryStorage() {
    const values = new Map();
    return {
        getItem: key => values.get(key) ?? null,
        setItem: (key, value) => values.set(key, value),
    };
}

test("saves and reloads a conversation", () => {
    const storage = memoryStorage();
    const chat = ChatHistory.create("chat-1", "How does habit stacking work?");
    chat.messages.push({ role: "user", text: "How does habit stacking work?" });
    chat.messages.push({ role: "assistant", text: "Pair a new habit with an existing one.", sources: [] });

    ChatHistory.save(storage, [chat]);

    assert.deepEqual(ChatHistory.load(storage), [chat]);
});

test("starts with no chats when storage is empty", () => {
    assert.deepEqual(ChatHistory.load(memoryStorage()), []);
});

test("rejects malformed or invalid saved chats", () => {
    const storage = memoryStorage();
    storage.setItem("atomic-habits-chats-v1", "not json");
    assert.throws(() => ChatHistory.load(storage));

    storage.setItem("atomic-habits-chats-v1", JSON.stringify([
        { id: "chat-1", title: "Bad chat", messages: [{ role: "admin", text: "unsafe" }] },
    ]));
    assert.throws(() => ChatHistory.load(storage), /invalid/);
});

test("uses the first question as the chat title", () => {
    const chat = ChatHistory.create("chat-2", "A".repeat(100));
    assert.equal(chat.title, "A".repeat(60));
    assert.deepEqual(chat.messages, []);
});
