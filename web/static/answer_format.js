const AnswerFormat = (() => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

    function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, character => entities[character]);
    }

    function sourcePage(source) {
        const page = Number(source?.metadata?.page_start);
        return Number.isSafeInteger(page) && page > 0 ? page : null;
    }

    function sourceTopic(source) {
        const metadata = source?.metadata || {};
        return String(metadata.section || metadata.chapter || "").trim() || "Atomic Habits";
    }

    function sourceLink(source, page = sourcePage(source)) {
        if (!Number.isSafeInteger(page) || page < 1) return null;
        const topic = escapeHtml(sourceTopic(source));
        return `href="/book.pdf#page=${page}" data-pdf-page="${page}" ` +
            `data-pdf-topic="${topic}" target="_blank" rel="noopener noreferrer"`;
    }

    function citationMatchesSource(source, chapter, page) {
        const metadata = source?.metadata || {};
        const chapterLabel = String(metadata.chapter || "").trim();
        const chapterMatch = /\bChapter\s+(\d+)\b/i.exec(chapterLabel);
        const chapterNumber = chapterMatch ? chapterMatch[1] : chapterLabel;
        const start = sourcePage(source);
        const end = Number(metadata.page_end);
        return chapterNumber === chapter && start !== null &&
            Number.isSafeInteger(end) && start <= page && page <= end;
    }

    function formatHtml(text, sources = []) {
        return String(text)
            .split(/\n\s*\n/)
            .filter(paragraph => paragraph.trim())
            .map(paragraph => {
                const escaped = escapeHtml(paragraph.trim());
                const headings = escaped.replace(/(^|\n)#{1,3}\s+([^\n]+)/g, "$1<strong>$2</strong>");
                const bold = headings.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
                const labels = bold.replace(/(^|\n)([^:\n]{3,140}:)(?=\s|$)/g, (match, start, label) =>
                    label.includes("<strong>") ? match : `${start}<strong>${label}</strong>`);
                const linked = labels.replace(/\(Chapter\s+(\d+),\s*p\.(\d+)\)/gi,
                    (citation, chapter, pageText) => {
                        const page = Number(pageText);
                        const source = (sources || []).find(source =>
                            citationMatchesSource(source, chapter, page));
                        if (!source) {
                            return citation;
                        }
                        return `<a class="citation-link" ${sourceLink(source, page)}>${citation}</a>`;
                    });
                return `<p>${linked}</p>`;
            })
            .join("");
    }

    return { escapeHtml, formatHtml, sourcePage, sourceTopic, sourceLink };
})();

if (typeof module !== "undefined") module.exports = AnswerFormat;
