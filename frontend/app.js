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

    // State variables
    let currentMode = "hybrid";
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
                "borda": "Borda Count Ranking"
            };

            const available = data.available_fusion_strategies || ["rrf", "weighted_sum", "combsum", "combmnz", "borda"];
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
        
        // Clear old results (remove cards, keep welcome card invisible)
        resultsList.querySelectorAll(".result-card, .welcome-card, .no-results-card").forEach(el => el.remove());

        // Gather request params
        const top_k = parseInt(topKSlider.value, 10);
        const strategy = fusionStrategySelect.value;
        const rrf_k = parseInt(rrfKSlider.value, 10);
        const fusion_alpha = parseFloat(fusionAlphaSlider.value);

        try {
            const res = await fetch(`${API_BASE}/api/search`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    query: query,
                    mode: currentMode,
                    top_k: top_k,
                    fusion_strategy: strategy,
                    rrf_k: rrf_k,
                    fusion_alpha: fusion_alpha
                })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Server error running search index");
            }

            const data = await res.json();
            
            // Hide loading indicator
            loadingIndicator.style.display = "none";

            // Render Metrics Dashboard & Results cards
            renderDashboard(data.latency);
            renderResults(data.results, query);

        } catch (err) {
            console.error("Search Failure:", err);
            loadingIndicator.style.display = "none";
            errorText.textContent = err.message || "An error occurred connecting to the backend retrieval server.";
            errorBox.style.display = "flex";
        }
    }

    // 8. Render Results list
    function renderResults(results, query) {
        if (!results || results.length === 0) {
            const noResultsCard = document.createElement("div");
            noResultsCard.className = "no-results-card glass-panel";
            noResultsCard.innerHTML = `
                <i class="fa-solid fa-face-frown no-results-icon"></i>
                <h3>No Relevant Passages Found</h3>
                <p>We searched over 8.8M passages but could not find anything matching "${escapeHtml(query)}". Try adjusting your query or lowering parameters.</p>
            `;
            resultsList.appendChild(noResultsCard);
            return;
        }

        results.forEach(doc => {
            const card = document.createElement("div");
            card.className = "result-card glass-panel";
            
            // Highlight text
            const highlightedText = highlightQueryTerms(doc.text, query);
            
            card.innerHTML = `
                <div class="result-bar"></div>
                <div class="result-header">
                    <div class="result-meta-left">
                        <span class="rank-badge">#${doc.rank}</span>
                        <button class="doc-id-btn" title="Copy Document ID" onclick="navigator.clipboard.writeText('${doc.doc_id}')">
                            ID: ${doc.doc_id} <i class="fa-regular fa-copy"></i>
                        </button>
                    </div>
                    <span class="score-badge">Score: ${parseFloat(doc.score).toFixed(4)}</span>
                </div>
                <div class="result-content">${highlightedText}</div>
            `;
            
            // Make the copy button interactive on click
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

            resultsList.appendChild(card);
        });
    }

    // 9. Update Latency indicators
    function renderDashboard(lat) {
        statsDashboard.style.display = "block";
        
        // Format strings
        updateStatCard(statTotal, lat.total_ms, "End-to-End Latency");
        updateStatCard(statSparse, currentMode !== "dense" ? lat.sparse_ms : 0.0, "Sparse Retrieval (BM25S)");
        updateStatCard(statDense, currentMode !== "sparse" ? lat.dense_ms : 0.0, "Dense Encoding & FAISS");
        updateStatCard(statFusion, currentMode === "hybrid" ? lat.fusion_ms : 0.0, "Fusion Processing");

        // Set relative progress fills
        const maxVal = Math.max(lat.total_ms, lat.sparse_ms, lat.dense_ms, lat.fusion_ms, 10);
        
        setFillWidth(statTotal, (lat.total_ms / maxVal) * 100);
        setFillWidth(statSparse, currentMode !== "dense" ? (lat.sparse_ms / maxVal) * 100 : 0);
        setFillWidth(statDense, currentMode !== "sparse" ? (lat.dense_ms / maxVal) * 100 : 0);
        setFillWidth(statFusion, currentMode === "hybrid" ? (lat.fusion_ms / maxVal) * 100 : 0);
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
        if (!text) return "";
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
