const test = require("node:test");
const assert = require("node:assert/strict");
const StreamEvents = require("../web/static/stream_events.js");

test("parses streamed answer and source events", () => {
    assert.deepEqual(StreamEvents.parse('data: {"type":"chunk","content":"Hello"}'), {
        type: "chunk", content: "Hello",
    });
    assert.deepEqual(StreamEvents.parse('data: {"type":"sources","sources":[]}'), {
        type: "sources", sources: [],
    });
    assert.equal(StreamEvents.parse(": heartbeat"), null);
    assert.equal(StreamEvents.parse("data: "), null);
});

test("rejects malformed or unexpected streamed events", () => {
    for (const line of [
        "data: {broken",
        "data: null",
        'data: {"type":"chunk","content":42}',
        'data: {"type":"sources","sources":{}}',
        'data: {"type":"unknown"}',
    ]) {
        assert.throws(() => StreamEvents.parse(line), /Invalid server event/);
    }
});
