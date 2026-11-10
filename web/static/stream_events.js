const StreamEvents = (() => {
    function parse(line) {
        if (!line.startsWith("data: ")) return null;
        const payload = line.slice(6).trim();
        if (!payload) return null;

        let event;
        try {
            event = JSON.parse(payload);
        } catch {
            throw new Error("Invalid server event");
        }

        if (!event || typeof event !== "object" || Array.isArray(event)) {
            throw new Error("Invalid server event");
        }
        if (event.type === "metadata" && typeof event.query_type === "string") return event;
        if (event.type === "chunk" && typeof event.content === "string") return event;
        if (event.type === "sources" && Array.isArray(event.sources)) return event;
        if (event.type === "error" && typeof event.message === "string") return event;
        if (event.type === "done" &&
            (event.answer === undefined || typeof event.answer === "string")) return event;
        throw new Error("Invalid server event");
    }

    return { parse };
})();

if (typeof module !== "undefined") module.exports = StreamEvents;
