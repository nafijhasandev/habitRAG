const ChatSettings = (() => {
    const storageKey = "atomic-habits-settings-v1";

    function load(storage) {
        const raw = storage.getItem(storageKey);
        if (!raw) return { showSources: true, theme: "dark" };

        const settings = JSON.parse(raw);
        const theme = settings?.theme ?? "dark";
        if (!settings || typeof settings.showSources !== "boolean" ||
            !["light", "dark"].includes(theme)) {
            throw new Error("Saved settings are invalid.");
        }
        return { showSources: settings.showSources, theme };
    }

    function save(storage, settings) {
        storage.setItem(storageKey, JSON.stringify(settings));
    }

    return { load, save };
})();

if (typeof module !== "undefined") module.exports = ChatSettings;
