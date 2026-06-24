// Interactive Application Script for VDT Hybrid Search
document.addEventListener("DOMContentLoaded", () => {
    // API Endpoint Base
    const API_BASE = "";

    // DOM Elements
    const queryInput = document.getElementById("query-input");
    const searchBtn = document.getElementById("search-btn");
    const queryChipsContainer = document.getElementById("query-chips");
    const modeBtns = document.querySelectorAll(".mode-btn");
    const advancedToggle = document.getElementById("advanced-toggle");
    const advancedSettings = document.getElementById("advanced-settings");
    
    // Config Sliders & Selects
    const topKSlider = document.getElementById("top-k-slider");
    const topKVal = document.getElementById("top-k-val");
    const fusionStrategySelect = document.getElementById("fusion-strategy");
    
    const rrfKGroup = document.getElementById("rrf-k-group");
    const rrfKSlider = document.getElementById("rrf-k-slider");
    const rrfKVal = document.getElementById("rrf-k-val");
    
    const fusionAlphaGroup = document.getElementById("fusion-alpha-group");
    const fusionAlphaSlider = document.getElementById("fusion-alpha-slider");
    const fusionAlphaVal = document.getElementById("fusion-alpha-val");

    const dateFromInput = document.getElementById("date-from");
    const dateToInput = document.getElementById("date-to");
    const clearDatesBtn = document.getElementById("clear-dates");

    // Rerank Elements
    const rerankToggle = document.getElementById("rerank-toggle");
    const rerankKSlider = document.getElementById("rerank-k-slider");
    const rerankKVal = document.getElementById("rerank-k-val");
    const wcrToggle = document.getElementById("wcr-toggle");
    const wcrAlphaSlider = document.getElementById("wcr-alpha-slider");
    const wcrAlphaVal = document.getElementById("wcr-alpha-val");
    const wcrSettings = document.getElementById("wcr-settings");

    // Latency Elements
    const statsDashboard = document.getElementById("stats-dashboard");
    const statTotal = document.getElementById("stat-total");
    const statSparse = document.getElementById("stat-sparse");
    const statDense = document.getElementById("stat-dense");
    const statFusion = document.getElementById("stat-fusion");

    // Results Containers & Statuses
    const resultsList = document.getElementById("results-list");
    const loadingIndicator = document.getElementById("loading-indicator");
    const errorBox = document.getElementById("error-box");
    const errorText = document.getElementById("error-text");

    // Pagination Elements
    const paginationControls = document.getElementById("pagination-controls");
    const pageInfo = document.getElementById("page-info");
    const pagePrevBtn = document.getElementById("page-prev");
    const pageNextBtn = document.getElementById("page-next");

    // Author Modal Elements
    const authorModal = document.getElementById("author-modal");
    const modalAuthorTitle = document.getElementById("modal-author-title");
    const modalBody = document.getElementById("modal-body");
    const modalCloseBtn = document.getElementById("modal-close");
    const modalPagination = document.getElementById("modal-pagination");
    const modalPageInfo = document.getElementById("modal-page-info");
    const modalPagePrevBtn = document.getElementById("modal-page-prev");
    const modalPageNextBtn = document.getElementById("modal-page-next");

    // State variables
    let currentMode = "hybrid";
    const RESULTS_PER_PAGE = 10;
    let allResults = [];      // All results from last search
    let currentPage = 1;
    let totalPages = 1;
    let lastQuery = "";

    // Author modal state
    let modalAuthorName = "";
    let modalCurrentPage = 1;
    let modalTotalPages = 1;

    const sampleQueries = [
        "Manhattan Project success",
        "What is reciprocal rank fusion",
        "hệ thống tìm kiếm kết hợp hybrid search",
        "dự án manhattan thành công",
        "large language models"
    ];

    // 1. Initialize Sample Chips
    renderSampleChips();
    
    function renderSampleChips() {
        queryChipsContainer.innerHTML = "";
        sampleQueries.forEach(q => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "query-chip";
            chip.textContent = q;
            chip.addEventListener("click", () => {
                queryInput.value = q;
                triggerSearch();
            });
            queryChipsContainer.appendChild(chip);
        });
    }

    // 2. Setup Mode Selectors
    modeBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            modeBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentMode = btn.getAttribute("data-mode");
            handleVisibilityRules();
        });
    });

    // 3. Collapsible Advanced Configurations Panel
    advancedToggle.addEventListener("click", () => {
        const isCollapsed = advancedSettings.classList.contains("collapsed");
        const caret = advancedToggle.querySelector(".caret-icon");
        
        if (isCollapsed) {
            advancedSettings.classList.remove("collapsed");
            caret.style.transform = "rotate(180deg)";
        } else {
            advancedSettings.classList.add("collapsed");
            caret.style.transform = "rotate(0deg)";
        }
    });

    // 4. Update Slider Value Display Bubbles
    topKSlider.addEventListener("input", (e) => {
        topKVal.textContent = e.target.value;
    });

    rrfKSlider.addEventListener("input", (e) => {
        rrfKVal.textContent = e.target.value;
    });

    fusionAlphaSlider.addEventListener("input", (e) => {
        fusionAlphaVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    rerankKSlider.addEventListener("input", (e) => {
        rerankKVal.textContent = e.target.value;
    });

    wcrAlphaSlider.addEventListener("input", (e) => {
        wcrAlphaVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    rerankToggle.addEventListener("change", (e) => {
        if (e.target.checked) {
            wcrSettings.style.display = "flex";
        } else {
            wcrSettings.style.display = "none";
        }
    });

    // 4b. Clear date filters button
    clearDatesBtn.addEventListener("click", () => {
        dateFromInput.value = "";
        dateToInput.value = "";
    });

    // 5. Connect to Backend & Populate Options on startup
    fetchEngineMetadata();

    async function fetchEngineMetadata() {
        try {
            const res = await fetch(`${API_BASE}/api/config`);
            if (!res.ok) throw new Error("Connection failed");
            const data = await res.json();
            
            // Populate fusion strategies select
            fusionStrategySelect.innerHTML = "";
            const strategyNames = {
                "rrf": "Reciprocal Rank Fusion (RRF)",
                "weighted_sum": "Weighted Sum of Scores",
                "combsum": "CombSUM Score Combination",
                "combmnz": "CombMNZ Multiplier",
                "borda": "Borda Count Ranking",
                "union": "Union Candidate Pool"
            };

            const available = data.available_fusion_strategies || ["rrf", "weighted_sum", "combsum", "combmnz", "borda", "union"];
            available.forEach(strategy => {
                const opt = document.createElement("option");
                opt.value = strategy;
                opt.textContent = strategyNames[strategy] || strategy.toUpperCase();
                fusionStrategySelect.appendChild(opt);
            });

            // Adjust parameters visibility after filling strategies
            handleVisibilityRules();

        } catch (err) {
            console.warn("Backend is offline or unreachable. Loading fallback default strategies.", err);
            // Load fallbacks
            fusionStrategySelect.innerHTML = `
                <option value="rrf">Reciprocal Rank Fusion (RRF)</option>
                <option value="weighted_sum">Weighted Sum of Scores</option>
                <option value="combsum">CombSUM Score Combination</option>
                <option value="combmnz">CombMNZ Multiplier</option>
                <option value="borda">Borda Count Ranking</option>
                <option value="union">Union Candidate Pool</option>
            `;
            handleVisibilityRules();
        }
    }

    // 6. Config Parameters Visibility toggles
    function handleVisibilityRules() {
        const hybridFields = document.querySelectorAll(".hybrid-only-field");
        
        if (currentMode === "hybrid") {
            hybridFields.forEach(el => el.style.display = "block");
            
            const strategy = fusionStrategySelect.value;
            if (strategy === "rrf") {
                rrfKGroup.style.display = "block";
                fusionAlphaGroup.style.display = "none";
            } else if (strategy === "weighted_sum") {
                rrfKGroup.style.display = "none";
                fusionAlphaGroup.style.display = "block";
            } else {
                rrfKGroup.style.display = "none";
                fusionAlphaGroup.style.display = "none";
            }
        } else {
            hybridFields.forEach(el => el.style.display = "none");
        }
    }

    fusionStrategySelect.addEventListener("change", handleVisibilityRules);

    // 7. Search Form Submission handlers
    searchBtn.addEventListener("click", triggerSearch);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            triggerSearch();
        }
    });

    async function triggerSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Reset UI states
        loadingIndicator.style.display = "flex";
        errorBox.style.display = "none";
        statsDashboard.style.display = "none";
        paginationControls.style.display = "none";
        
        // Clear old results (remove cards, keep welcome card invisible)
        resultsList.querySelectorAll(".result-card, .welcome-card, .no-results-card, .filter-info-card").forEach(el => el.remove());

        // Gather request params
        const top_k = parseInt(topKSlider.value, 10);
        const strategy = fusionStrategySelect.value;
        const rrf_k = parseInt(rrfKSlider.value, 10);
        const fusion_alpha = parseFloat(fusionAlphaSlider.value);

        // Date filter params
        const date_from = dateFromInput.value || null;
        const date_to = dateToInput.value || null;

        try {
            const reqBody = {
                query: query,
                mode: currentMode,
                top_k: top_k,
                fusion_strategy: strategy,
                rrf_k: rrf_k,
                fusion_alpha: fusion_alpha,
                rerank: rerankToggle.checked,
                rerank_top_k: parseInt(rerankKSlider.value, 10),
                wcr: wcrToggle ? wcrToggle.checked : false,
                wcr_alpha: wcrAlphaSlider ? parseFloat(wcrAlphaSlider.value) : 0.5,
            };
            if (date_from) reqBody.date_from = date_from;
            if (date_to) reqBody.date_to = date_to;

            const res = await fetch(`${API_BASE}/api/search`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(reqBody)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Server error running search index");
            }

            const data = await res.json();
            
            // Hide loading indicator
            loadingIndicator.style.display = "none";

            // Render Metrics Dashboard
            renderDashboard(data.latency);

            // Store all results for pagination
            allResults = data.results;
            lastQuery = query;
            currentPage = 1;
            totalPages = Math.max(1, Math.ceil(allResults.length / RESULTS_PER_PAGE));

            // Show filter info if date filter was active
            if ((date_from || date_to) && data.total_before_filter > data.num_results) {
                const filterCard = document.createElement("div");
                filterCard.className = "filter-info-card glass-panel";
                const fromStr = date_from || "∞ past";
                const toStr = date_to || "∞ future";
                filterCard.innerHTML = `
                    <i class="fa-solid fa-filter"></i>
                    <span>Showing <strong>${data.num_results}</strong> of <strong>${data.total_before_filter}</strong> retrieved passages (filtered by date: ${fromStr} → ${toStr})</span>
                `;
                resultsList.appendChild(filterCard);
            }

            renderCurrentPage();

        } catch (err) {
            console.error("Search Failure:", err);
            loadingIndicator.style.display = "none";
            errorText.textContent = err.message || "An error occurred connecting to the backend retrieval server.";
            errorBox.style.display = "flex";
        }
    }

    // 8. Pagination Logic
    function renderCurrentPage() {
        // Remove old result cards only (keep filter-info-card)
        resultsList.querySelectorAll(".result-card, .no-results-card").forEach(el => el.remove());

        if (!allResults || allResults.length === 0) {
            const noResultsCard = document.createElement("div");
            noResultsCard.className = "no-results-card glass-panel";
            noResultsCard.innerHTML = `
                <i class="fa-solid fa-face-frown no-results-icon"></i>
                <h3>No Relevant Passages Found</h3>
                <p>We searched over 8.8M passages but could not find anything matching "${escapeHtml(lastQuery)}". Try adjusting your query or lowering parameters.</p>
            `;
            resultsList.appendChild(noResultsCard);
            paginationControls.style.display = "none";
            return;
        }

        const startIdx = (currentPage - 1) * RESULTS_PER_PAGE;
        const endIdx = Math.min(startIdx + RESULTS_PER_PAGE, allResults.length);
        const pageResults = allResults.slice(startIdx, endIdx);

        pageResults.forEach(doc => {
            const card = createResultCard(doc, lastQuery);
            resultsList.appendChild(card);
        });

        // Update pagination controls
        if (totalPages > 1) {
            paginationControls.style.display = "flex";
            pageInfo.textContent = `Page ${currentPage} / ${totalPages}  (${allResults.length} results)`;
            pagePrevBtn.disabled = currentPage <= 1;
            pageNextBtn.disabled = currentPage >= totalPages;
        } else {
            paginationControls.style.display = "none";
        }
    }

    pagePrevBtn.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            renderCurrentPage();
            scrollToResults();
        }
    });

    pageNextBtn.addEventListener("click", () => {
        if (currentPage < totalPages) {
            currentPage++;
            renderCurrentPage();
            scrollToResults();
        }
    });

    function scrollToResults() {
        resultsList.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // 9. Create a result card element
    function createResultCard(doc, query) {
        const card = document.createElement("div");
        card.className = "result-card glass-panel";
        
        // Highlight text
        const highlightedText = highlightQueryTerms(doc.text, query);

        // Build metadata line with clickable author
        let metaHtml = "";
        if (doc.author_name || doc.written_date) {
            const authorPart = doc.author_name
                ? `<a class="meta-item author-link" href="#" data-author="${escapeHtml(doc.author_name)}"><i class="fa-solid fa-user-pen"></i> ${escapeHtml(doc.author_name)}</a>`
                : "";
            const datePart = doc.written_date
                ? `<span class="meta-item"><i class="fa-regular fa-calendar"></i> ${escapeHtml(doc.written_date)}</span>`
                : "";
            metaHtml = `<div class="result-metadata">${authorPart}${datePart}</div>`;
        }

        // Score badge: only show if score field exists (search results)
        const scoreBadge = doc.score !== undefined
            ? `<span class="score-badge">Score: ${parseFloat(doc.score).toFixed(4)}</span>`
            : "";

        // Rank badge
        const rankBadge = doc.rank !== undefined
            ? `<span class="rank-badge">#${doc.rank}</span>`
            : "";
        
        card.innerHTML = `
            <div class="result-bar"></div>
            <div class="result-header">
                <div class="result-meta-left">
                    ${rankBadge}
                    <button class="doc-id-btn" title="Copy Document ID">
                        ID: ${doc.doc_id} <i class="fa-regular fa-copy"></i>
                    </button>
                </div>
                ${scoreBadge}
            </div>
            ${metaHtml}
            <div class="result-content">${highlightedText}</div>
        `;
        
        // Copy button
        const copyBtn = card.querySelector(".doc-id-btn");
        copyBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(doc.doc_id);
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = `Copied! <i class="fa-solid fa-check" style="color: var(--accent-total)"></i>`;
            copyBtn.style.borderColor = "var(--accent-total)";
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
                copyBtn.style.borderColor = "";
            }, 1200);
        });

        // Author link click handler
        const authorLink = card.querySelector(".author-link");
        if (authorLink) {
            authorLink.addEventListener("click", (e) => {
                e.preventDefault();
                const name = authorLink.getAttribute("data-author");
                openAuthorModal(name);
            });
        }

        return card;
    }

    // 10. Author Modal
    function openAuthorModal(authorName) {
        modalAuthorName = authorName;
        modalCurrentPage = 1;
        authorModal.style.display = "flex";
        document.body.style.overflow = "hidden";
        loadAuthorPassages(authorName, 1);
    }

    function closeAuthorModal() {
        authorModal.style.display = "none";
        document.body.style.overflow = "";
        modalBody.innerHTML = "";
        modalPagination.style.display = "none";
    }

    modalCloseBtn.addEventListener("click", closeAuthorModal);
    authorModal.addEventListener("click", (e) => {
        if (e.target === authorModal) closeAuthorModal();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && authorModal.style.display === "flex") {
            closeAuthorModal();
        }
    });

    async function loadAuthorPassages(authorName, page) {
        modalAuthorTitle.innerHTML = `<i class="fa-solid fa-user-pen"></i> ${escapeHtml(authorName)}`;
        modalBody.innerHTML = `
            <div class="modal-loading">
                <div class="spinner"></div>
                <p>Loading passages...</p>
            </div>
        `;
        modalPagination.style.display = "none";

        try {
            const params = new URLSearchParams({
                author_name: authorName,
                page: page,
                page_size: 20,
            });
            const res = await fetch(`${API_BASE}/api/author/passages?${params}`);
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Failed to load author passages");
            }

            const data = await res.json();
            modalCurrentPage = data.page;
            modalTotalPages = data.total_pages;

            modalBody.innerHTML = "";

            if (data.passages.length === 0) {
                modalBody.innerHTML = `
                    <div class="modal-empty">
                        <i class="fa-solid fa-inbox"></i>
                        <p>No passages found for this author.</p>
                    </div>
                `;
                return;
            }

            // Summary info
            const summaryEl = document.createElement("div");
            summaryEl.className = "modal-summary";
            summaryEl.innerHTML = `<strong>${data.total.toLocaleString()}</strong> passages found for <strong>${escapeHtml(authorName)}</strong>`;
            modalBody.appendChild(summaryEl);

            // Render passage cards
            data.passages.forEach((passage, idx) => {
                const passageCard = document.createElement("div");
                passageCard.className = "modal-passage-card";

                const globalIdx = (data.page - 1) * data.page_size + idx + 1;

                passageCard.innerHTML = `
                    <div class="modal-passage-header">
                        <span class="modal-passage-num">#${globalIdx}</span>
                        <span class="modal-passage-docid">ID: ${passage.doc_id}</span>
                        ${passage.written_date ? `<span class="modal-passage-date"><i class="fa-regular fa-calendar"></i> ${escapeHtml(passage.written_date)}</span>` : ""}
                    </div>
                    <div class="modal-passage-text">${escapeHtml(passage.text)}</div>
                `;
                modalBody.appendChild(passageCard);
            });

            // Pagination
            if (modalTotalPages > 1) {
                modalPagination.style.display = "flex";
                modalPageInfo.textContent = `Page ${modalCurrentPage} / ${modalTotalPages}`;
                modalPagePrevBtn.disabled = modalCurrentPage <= 1;
                modalPageNextBtn.disabled = modalCurrentPage >= modalTotalPages;
            } else {
                modalPagination.style.display = "none";
            }

        } catch (err) {
            console.error("Author passages error:", err);
            modalBody.innerHTML = `
                <div class="modal-empty">
                    <i class="fa-solid fa-triangle-exclamation" style="color: hsl(0, 80%, 55%);"></i>
                    <p>${escapeHtml(err.message)}</p>
                </div>
            `;
        }
    }

    modalPagePrevBtn.addEventListener("click", () => {
        if (modalCurrentPage > 1) {
            loadAuthorPassages(modalAuthorName, modalCurrentPage - 1);
        }
    });

    modalPageNextBtn.addEventListener("click", () => {
        if (modalCurrentPage < modalTotalPages) {
            loadAuthorPassages(modalAuthorName, modalCurrentPage + 1);
        }
    });

    // 11. Update Latency indicators
    function renderDashboard(lat) {
        statsDashboard.style.display = "block";
        
        // Format strings
        updateStatCard(statTotal, lat.total_ms, "End-to-End Latency");
        updateStatCard(statSparse, currentMode !== "dense" ? lat.sparse_ms : 0.0, "Sparse Retrieval (BM25S)");
        updateStatCard(statDense, currentMode !== "sparse" ? lat.dense_ms : 0.0, "Dense Encoding & FAISS");
        updateStatCard(statFusion, currentMode === "hybrid" ? lat.fusion_ms : 0.0, "Fusion Processing");

        const statRerank = document.getElementById("stat-rerank");
        if (lat.rerank_ms !== undefined && lat.rerank_ms !== null) {
            statRerank.style.display = "flex";
            updateStatCard(statRerank, lat.rerank_ms, "CrossEncoder Rerank");
        } else {
            statRerank.style.display = "none";
        }

        // Set relative progress fills
        const maxVal = Math.max(lat.total_ms, lat.sparse_ms || 0, lat.dense_ms || 0, lat.fusion_ms || 0, lat.rerank_ms || 0, 10);
        
        setFillWidth(statTotal, (lat.total_ms / maxVal) * 100);
        setFillWidth(statSparse, currentMode !== "dense" ? (lat.sparse_ms / maxVal) * 100 : 0);
        setFillWidth(statDense, currentMode !== "sparse" ? (lat.dense_ms / maxVal) * 100 : 0);
        setFillWidth(statFusion, currentMode === "hybrid" ? (lat.fusion_ms / maxVal) * 100 : 0);
        setFillWidth(statRerank, (lat.rerank_ms !== undefined && lat.rerank_ms !== null) ? (lat.rerank_ms / maxVal) * 100 : 0);
    }

    function updateStatCard(cardElement, val, label) {
        const valText = cardElement.querySelector(".stat-value");
        valText.textContent = `${parseFloat(val).toFixed(1)} ms`;
        if (parseFloat(val) === 0.0) {
            cardElement.style.opacity = "0.35";
        } else {
            cardElement.style.opacity = "1";
        }
    }

    function setFillWidth(cardElement, percentage) {
        const fill = cardElement.querySelector(".fill");
        fill.style.width = `${percentage}%`;
    }

    // Highlighting logic
    function highlightQueryTerms(text, query) {
        if (!text || !query) return escapeHtml(text || "");
        const terms = query
            .toLowerCase()
            .split(/[\s,.\-\/]+/)
            .filter(t => t.length >= 3)
            .filter((value, index, self) => self.indexOf(value) === index);
            
        if (terms.length === 0) return escapeHtml(text);
        
        // Escape characters for regex
        const escapedTerms = terms.map(t => t.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
        const pattern = new RegExp(`\\b(${escapedTerms.join('|')})\\b`, 'gi');
        
        // Perform replacement safely on text
        const safeText = escapeHtml(text);
        try {
            return safeText.replace(pattern, '<mark class="query-highlight">$1</mark>');
        } catch (e) {
            return safeText;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.innerText = text;
        return div.innerHTML;
    }
});
