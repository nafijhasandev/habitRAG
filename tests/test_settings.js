const test = require("node:test");
const assert = require("node:assert/strict");
const ChatSettings = require("../web/static/settings.js");

function memoryStorage() {
    const values = new Map();
    return {
        getItem: key => values.get(key) ?? null,
        setItem: (key, value) => values.set(key, value),
    };
}

test("shows sources by default", () => {
    assert.deepEqual(ChatSettings.load(memoryStorage()), { showSources: true, theme: "dark" });
});

test("saves and reloads source and color preferences", () => {
    const storage = memoryStorage();
    ChatSettings.save(storage, { showSources: false, theme: "light" });
    assert.deepEqual(ChatSettings.load(storage), { showSources: false, theme: "light" });
});

test("loads older settings without a color preference", () => {
    const storage = memoryStorage();
    ChatSettings.save(storage, { showSources: false });
    assert.deepEqual(ChatSettings.load(storage), { showSources: false, theme: "dark" });
});

test("rejects malformed settings", () => {
    const storage = memoryStorage();
    storage.setItem("atomic-habits-settings-v1", "not json");
    assert.throws(() => ChatSettings.load(storage));

    storage.setItem("atomic-habits-settings-v1", '{"showSources":"no"}');
    assert.throws(() => ChatSettings.load(storage), /invalid/);

    storage.setItem("atomic-habits-settings-v1", '{"showSources":true,"theme":"blue"}');
    assert.throws(() => ChatSettings.load(storage), /invalid/);
});
