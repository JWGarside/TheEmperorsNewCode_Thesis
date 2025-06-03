// paper_code_compare_batch.js
const fs = require('fs');
const path = require('path');
const { Anthropic } = require('@anthropic-ai/sdk');
const { program } = require('commander');
const { v4: uuidv4 } = require('uuid');
const semanticChunking = require('./semanticChunking');

/**
 * Compare a research paper and code files/README from a zip archive using a two-stage Claude API process.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} zipPath - Path to the zip archive containing code and README
 * @param {string} outputDir - Directory to save the output file
 * @param {number} runNumber - The run number for this comparison
 * @param {string} customPrompt - Custom prompt for the second stage (code comparison)
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<string>} - Claude's final response
 */
async function comparePaperAndCode(pdfPath, zipPath, outputDir, runNumber, customPrompt = null, apiKey = null) {
    // Initialize the Anthropic client
    const anthropic = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY || "your-anthropic-key-here"
    });

    // Read and encode PDF file as Base64
    const pdfData = fs.readFileSync(pdfPath).toString('base64');

    // Semantically chunk the zip archive
    const { codeFiles, readmeContent } = await semanticChunking.chunkZipArchive(zipPath);

    // Extract file names for output naming
    const pdfName = path.basename(pdfPath, path.extname(pdfPath));
    const zipName = path.basename(zipPath, path.extname(zipPath));

    // --- Stage 1: Extract Paper Details ---
    const paperAnalysisPrompt = (
        "Please analyze the provided research paper (PDF document). Your primary goal is to identify and summarize the paper's core technical claims and contributions.\n\n" +
        "Specifically, identify and list:\n" +
        "1.  **Core Claims/Contributions:** What are the main scientific or technical advancements presented in the paper?\n" +
        "2.  **Key Methodological Details:**\n" +
        "    *   Algorithms used or proposed.\n" +
        "    *   Specific model architectures (if applicable).\n" +
        "    *   Important parameter values or configurations explicitly stated in the paper.\n" +
        "    *   Datasets used for experiments.\n" +
        "    *   Evaluation metrics.\n\n" +
        "    *   Identify which aspects are presented as fundamental to the approach versus optimization choices.\n\n" +
        "Present this information in a structured and clear manner. This output will be used to compare against a codebase. Ensure the output is comprehensive."
    );

    const paperAnalysisConversationId = uuidv4();
    console.log(`Stage 1: Analyzing paper "${pdfName}" to extract key details...`);
    
    let extractedPaperDetails = "";
    try {
        const paperResponse = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219", // Or your preferred model for summarization
            max_tokens: 1024, // Standard max_tokens for this stage
            temperature: 0.3, // Lower temperature for factual extraction
            system: "This is a new conversation with ID: " + paperAnalysisConversationId,
            messages: [
                {
                    role: "user",
                    content: [
                        {
                            type: "document",
                            source: {
                                type: "base64",
                                media_type: "application/pdf",
                                data: pdfData
                            }
                        },
                        {
                            type: "text",
                            text: paperAnalysisPrompt
                        }
                    ]
                }
            ]
        });
        extractedPaperDetails = paperResponse.content[0].text;
        console.log("Stage 1: Paper details extracted successfully.");
        // Optionally, save intermediate results for debugging:
        // fs.writeFileSync(path.join(outputDir, `${pdfName}_extracted_details_Run${runNumber}.txt`), extractedPaperDetails);

    } catch (error) {
        console.error('Error in Stage 1 (Paper Analysis):', error);
        if (error.response) {
            console.error('API Error Details:', error.response.data);
        }
        throw error;
    }

    // --- Stage 2: Compare Code with Extracted Paper Details ---
    const codeComparisonPrompt = customPrompt || (
        "# Research Code Reproducibility Analysis (Two-Stage)\n\n" +
        "You have been provided with:\n" +
        "1. A summary of core claims and key technical details extracted from a research paper (under 'EXTRACTED PAPER DETAILS').\n" +
        "2. The codebase (and README) associated with that paper.\n\n" +
        "Your task is to analyze the codebase in light of the provided paper details and assess reproducibility.\n\n" +
        "## Analysis Steps:\n" +
        "1.  **Review Paper Details:** Carefully read the provided summary of the paper's core claims and key methodological details.\n" +
        "2.  **Examine Code Implementation:**\n" +
        "    *   Examine how the core algorithms, architectures, and methodologies (as described in the provided paper details) are implemented in the code.\n" +
        "    *   Trace the execution flow of key components.\n" +
        "    *   Note any parameters, constants, or design choices in the code that correspond to or deviate from the paper's description. Think step-by-step when analyzing the code.\n" +
        "3.  **Identify Discrepancies:** Compare the code implementation against the provided paper details. Note any discrepancies.\n\n" +
        "## Discrepancy Classification:\n" +
        "Classify discrepancies as:\n" +
        "-   **Critical**: Prevent reproduction of core claims/methodology as described in the paper details.\n" +
        "-   **Minor**: May affect performance or specific results but do not alter the fundamental approach described.\n" +
        "-   **Cosmetic**: Differences in documentation, code style, or variable names with minimal impact on reproducibility or understanding.\n\n" +
        "## Output Format:\n" +
        "1.  **Brief Paper Summary and Core Claims (Recap):** Briefly reiterate the core claims and methodology based *on the provided summary*.\n" +
        "2.  **Implementation Assessment:** Describe how the code implements (or fails to implement) the key aspects outlined in the provided paper details.\n" +
        "3.  **Categorized Discrepancies:** List any identified discrepancies, categorized as Critical, Minor, or Cosmetic. For each, explain the discrepancy and reference relevant parts of the code and the provided paper details.\n" +
        "4.  **Overall Reproducibility Conclusion:** Based on your analysis, conclude on the reproducibility of the work. If no significant discrepancies are found that would impact reproducibility, state this clearly.\n\n" +
        "Focus on whether the code implementation preserves the fundamental approach and claims outlined in the provided paper details."
    );

    const codeComparisonConversationId = uuidv4();
    console.log(`Stage 2: Comparing code from "${zipName}" with extracted paper details...`);

    try {
        const contentForCodeComparison = [
            {
                type: "text",
                text: "EXTRACTED PAPER DETAILS:\n\n" + extractedPaperDetails + "\n\n---\n\nCODE FILES AND README:\n\n"
            }
        ];

        // Add each code file
        for (const file of codeFiles) {
            contentForCodeComparison.push({
                type: "text",
                text: `File: ${file.name}\n\`\`\`${file.language}\n${file.content}\n\`\`\`\n`
            });
        }
        
        // Add the README content
        contentForCodeComparison.push({
            type: "text", 
            text: `README:\n${readmeContent}\n\n---\n\nANALYSIS TASK:\n\n${codeComparisonPrompt}`
        });

        // Send request to Claude API for code comparison
        const response = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219",
            max_tokens: 2048, // Extended tokens for the main analysis
            temperature: 0.5,
            system: "This is a new conversation with ID: " + codeComparisonConversationId,
            messages: [
                {
                    role: "user",
                    content: contentForCodeComparison
                }
            ] 
        });

        // Get Claude's response text
        const result = response.content[0].text;

        // Save result to markdown file  
        const outputFileName = `${zipName}_Run${runNumber}_TwoStage.md`; // Added _TwoStage to differentiate
        const outputPath = path.join(outputDir, outputFileName);

        // Create markdown content
        const mdContent = `# Paper-Code Consistency Analysis (Two-Stage)

**Paper:** ${pdfName}  
**Code Archive:** ${zipName}
**Analysis Date:** ${new Date().toISOString().split('T')[0]}

## Extracted Paper Details (Stage 1 Output)
\`\`\`text
${extractedPaperDetails}
\`\`\`

## Analysis Results (Stage 2 Output)

${result}`;

        fs.writeFileSync(outputPath, mdContent);

        console.log(`Stage 2: Analysis saved to: ${outputPath}`);
        return result;
    } catch (error) {
        console.error('Error in Stage 2 (Code Comparison):', error);
        if (error.response) {
            console.error('API Error Details:', error.response.data);
        }
        throw error; 
    }
}

/**
 * Run a batch of comparisons between a paper and code zip archive multiple times.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} zipPath - Path to the zip archive containing code and README
 * @param {string} outputDir - Directory to save the output file
 * @param {number} numRuns - Number of times to run each comparison (default: 3)
 * @param {string} customPrompt - Custom prompt for Claude (uses default if null)
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<void>} 
 */
async function runComparisonsInBatch(pdfPath, zipPath, outputDir, numRuns = 3, customPrompt = null, apiKey = null) {
    for (let i = 1; i <= numRuns; i++) {
        console.log(`Running comparison ${i} of ${numRuns}...`);
        await comparePaperAndCode(pdfPath, zipPath, outputDir, i, customPrompt, apiKey);
    }
}

// Command line interface setup
if (require.main === module) {
    program
        .name('paper-code-compare-batch')
        .description('Compare a research paper and code zip archive with Claude AI in batch mode') 
        .version('1.0.0')
        .requiredOption('-p, --pdf <path>', 'Path to the PDF research paper')
        .requiredOption('-z, --zip <path>', 'Path to the zip archive containing code and README') 
        .option('-o, --output-dir <path>', 'Directory to save output files', './CS2 Results Output/CS2 PaperThenCode results') // Configure this to your desired folder
        .option('-r, --runs <number>', 'Number of times to run each comparison', 3) // Default to 3 runs
        .option('--prompt <text>', 'Custom prompt for Claude (uses default if not provided)')
        .option('--api-key <key>', 'Anthropic API key (optional, can use environment variable)')
        .action(async (options) => {
            try {
                // Ensure output directory exists
                if (!fs.existsSync(options.outputDir)) {
                    fs.mkdirSync(options.outputDir, { recursive: true }); 
                }

                // Run the comparisons in batch
                await runComparisonsInBatch(
                    options.pdf,
                    options.zip,  
                    options.outputDir,
                    options.runs,
                    options.prompt,
                    options.apiKey
                );

                console.log(`\nAll comparisons completed. Results saved in ${options.outputDir}`); 
            } catch (error) {
                console.error(`Error: ${error.message}`);
                process.exit(1);
            }
        });

    program.parse();  
}

module.exports = { comparePaperAndCode, runComparisonsInBatch };