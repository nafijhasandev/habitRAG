const test = require("node:test");
const assert = require("node:assert/strict");
const AnswerFormat = require("../web/static/answer_format.js");

test("renders bold topic names and chapter citations", () => {
    assert.equal(
        AnswerFormat.formatHtml("**Habit stacking** builds on a cue.\n\nSee **Chapter 5**."),
        "<p><strong>Habit stacking</strong> builds on a cue.</p><p>See <strong>Chapter 5</strong>.</p>",
    );
});

test("renders markdown headings as bold text", () => {
    assert.equal(
        AnswerFormat.formatHtml("## The Four Laws\nMake it obvious."),
        "<p><strong>The Four Laws</strong>\nMake it obvious.</p>",
    );
});

test("bolds plain topic introductions and numbered labels", () => {
    assert.equal(
        AnswerFormat.formatHtml("The four laws are:\n\n1. The first law (Cue): Make it obvious."),
        "<p><strong>The four laws are:</strong></p><p><strong>1. The first law (Cue):</strong> Make it obvious.</p>",
    );
});

test("escapes model output before adding formatting", () => {
    assert.equal(
        AnswerFormat.formatHtml("**<script>alert('x')</script>** & details"),
        "<p><strong>&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;</strong> &amp; details</p>",
    );
});

test("links only citations supported by retrieved PDF pages", () => {
    const sources = [{ metadata: {
        chapter: "Chapter 5: The Best Way to Start a New Habit",
        section: "HABIT STACKING",
        page_start: 73,
        page_end: 75,
    } }];
    assert.equal(
        AnswerFormat.formatHtml("Use a cue (Chapter 5, p.74). Avoid guesses (Chapter 9, p.99).", sources),
        '<p>Use a cue <a class="citation-link" href="/book.pdf#page=74" data-pdf-page="74" data-pdf-topic="HABIT STACKING" target="_blank" rel="noopener noreferrer">(Chapter 5, p.74)</a>. Avoid guesses (Chapter 9, p.99).</p>',
    );
});

test("source links carry a safe topic and page for the side viewer", () => {
    const source = { metadata: {
        section: 'A "quoted" <topic>', page_start: 91,
    } };
    assert.equal(AnswerFormat.sourceTopic(source), 'A "quoted" <topic>');
    assert.equal(
        AnswerFormat.sourceLink(source),
        'href="/book.pdf#page=91" data-pdf-page="91" data-pdf-topic="A &quot;quoted&quot; &lt;topic&gt;" target="_blank" rel="noopener noreferrer"',
    );
    assert.equal(AnswerFormat.sourceLink({ metadata: { page_start: -1 } }), null);
    assert.equal(AnswerFormat.sourceTopic({ metadata: { section: "  " } }), "Atomic Habits");
});

test("rejects invalid source pages and keeps answer HTML escaped", () => {
    const sources = [{ metadata: { chapter: "Chapter 5", page_start: -1, page_end: 75 } }];
    assert.equal(AnswerFormat.sourcePage(sources[0]), null);
    assert.equal(
        AnswerFormat.formatHtml("<script>alert(1)</script> (Chapter 5, p.74)", sources),
        "<p>&lt;script&gt;alert(1)&lt;/script&gt; (Chapter 5, p.74)</p>",
    );
});
