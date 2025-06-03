// paper_code_compare_batch.js
const fs = require('fs');
const path = require('path');
const { Anthropic } = require('@anthropic-ai/sdk');
const { program } = require('commander');
const { v4: uuidv4 } = require('uuid');
const semanticChunking = require('./semanticChunking');

/**
 * Compare a research paper and code files/README from a zip archive using a two-stage Claude API process
 * (Code analysis first, then paper comparison).
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} zipPath - Path to the zip archive containing code and README
 * @param {string} outputDir - Directory to save the output file
 * @param {number} runNumber - The run number for this comparison
 * @param {string} customPrompt - Custom prompt for the second stage (paper and code summary comparison)
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

    // --- Stage 1: Analyze Codebase ---
    const codeAnalysisPrompt = (
        "You are provided with the extracted contents (code files and README) from a codebase archive. Your primary goal is to perform a thorough analysis of this codebase and generate a detailed summary.\n\n" +
        "Specifically, identify and describe:\n" +
        "1.  **Overall Structure and Organization:** How is the project structured? What are the main components, modules, or directories and their apparent roles?\n" +
        "2.  **Core Functionality:** What are the primary functions or capabilities of this codebase? What problem does it seem to solve or what tasks does it perform?\n" +
        "3.  **Key Algorithms and Logic:** Describe any significant algorithms, data processing pipelines, or complex logic flows you can identify within the code.\n" +
        "4.  **Important Parameters and Configurations:** Note any key parameters, constants, configuration settings, or dependencies that appear crucial to the codebase's operation (as evident from the code or README).\n" +
        "5.  **Data Handling:** How does the code appear to handle input and output data? What data formats or structures are used?\n" +
        "6.  **README Summary:** Briefly summarize the key information provided in the README, if available (e.g., project purpose, setup, usage).\n\n" +
        "Present this information in a structured and clear manner. This output will be used as a detailed reference for comparing against a research paper. Ensure the output is comprehensive and accurately reflects the provided code."
    );

    const codeAnalysisConversationId = uuidv4();
    console.log(`Stage 1: Analyzing codebase from "${zipName}" to extract key details...`);
    
    let extractedCodeDetails = "";
    try {
        const contentForCodeAnalysis = [];
        contentForCodeAnalysis.push({ type: "text", text: `README Content:\n${readmeContent}\n\n---\n\nCODE FILES:\n` });
        for (const file of codeFiles) {
            contentForCodeAnalysis.push({
                type: "text",
                text: `File: ${file.name}\n\`\`\`${file.language}\n${file.content}\n\`\`\`\n`
            });
        }
        contentForCodeAnalysis.push({ type: "text", text: `\n\n---\n\nANALYSIS TASK:\n\n${codeAnalysisPrompt}` });

        const codeResponse = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219", 
            max_tokens: 3072, // Allow more tokens for detailed code summary
            temperature: 0.3, 
            system: "This is a new conversation with ID: " + codeAnalysisConversationId,
            messages: [
                {
                    role: "user",
                    content: contentForCodeAnalysis
                }
            ]
        });
        extractedCodeDetails = codeResponse.content[0].text;
        console.log("Stage 1: Codebase details extracted successfully.");
        // Optionally, save intermediate results for debugging:
        // fs.writeFileSync(path.join(outputDir, `${zipName}_extracted_code_details_Run${runNumber}.txt`), extractedCodeDetails);

    } catch (error) {
        console.error('Error in Stage 1 (Codebase Analysis):', error);
        if (error.response) {
            console.error('API Error Details:', error.response.data);
        }
        throw error;
    }

    // --- Stage 2: Analyze Paper and Compare with Extracted Code Details ---
    const paperAndCodeComparisonPrompt = customPrompt || (
        "# Research Paper and Codebase Consistency Analysis (Code-First Two-Stage)\n\n" +
        "You have been provided with:\n" +
        "1. A detailed summary of a codebase (under 'EXTRACTED CODEBASE DETAILS').\n" +
        "2. A research paper (PDF document) that is supposedly related to this codebase.\n\n" +
        "Your task is to:\n" +
        "A. Analyze the research paper to understand its core claims, methodology, algorithms, and key technical details.\n" +
        "B. Compare these findings from the paper against the provided 'EXTRACTED CODEBASE DETAILS'.\n" +
        "C. Assess the reproducibility and identify discrepancies.\n\n" +
        "## Analysis Steps:\n" +
        "1.  **Analyze Research Paper:**\n" +
        "    *   Identify the paper's core claims and contributions.\n" +
        "    *   Note key methodological details: algorithms, model architectures, important parameters, datasets, evaluation metrics.\n" +
        "    *   Pay attention to information in figures and tables if they describe crucial aspects.\n" +
        "2.  **Compare Paper with Codebase Summary:**\n" +
        "    *   Carefully compare the paper's technical descriptions with the 'EXTRACTED CODEBASE DETAILS'.\n" +
        "    *   Does the codebase summary align with what the paper describes? Where does it differ?\n" +
        "3.  **Identify Discrepancies:** Based on the comparison, note any discrepancies between the paper's description and the codebase's summarized implementation.\n\n" +
        "## Discrepancy Classification:\n" +
        "Classify discrepancies as:\n" +
        "-   **Critical**: Prevent reproduction of core claims/methodology as described in the paper, based on the codebase summary.\n" +
        "-   **Minor**: May affect performance or specific results but do not alter the fundamental approach described in the paper, according to the codebase summary.\n" +
        "-   **Cosmetic**: Differences in terminology or minor details with minimal impact on reproducibility or understanding, when comparing paper to codebase summary.\n\n" +
        "## Output Format:\n" +
        "1.  **Brief Paper Summary:** Summarize the paper's core claims, methodology, and key technical findings.\n" +
        "2.  **Implementation Assessment (based on Codebase Summary vs. Paper):** Describe how the 'EXTRACTED CODEBASE DETAILS' align with or deviate from the paper's descriptions.\n" +
        "3.  **Categorized Discrepancies:** List any identified discrepancies, categorized as Critical, Minor, or Cosmetic. For each, explain the discrepancy, referencing relevant parts of the paper and the 'EXTRACTED CODEBASE DETAILS'.\n" +
        "4.  **Overall Reproducibility Conclusion:** Based on your analysis, conclude on the reproducibility of the work. If no significant discrepancies are found that would impact reproducibility, state this clearly.\n\n" +
        "Focus on whether the codebase, as summarized, implements the fundamental approach and claims of the paper."
    );

    const paperComparisonConversationId = uuidv4();
    console.log(`Stage 2: Analyzing paper "${pdfName}" and comparing with extracted codebase details from "${zipName}"...`);

    try {
        const contentForPaperComparison = [
            {
                type: "text",
                text: "EXTRACTED CODEBASE DETAILS:\n\n" + extractedCodeDetails + "\n\n---\n\nRESEARCH PAPER (PDF) AND ANALYSIS TASK:\n\n"
            },
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
                text: `\n\n---\n\n${paperAndCodeComparisonPrompt}`
            }
        ];

        // Send request to Claude API for paper analysis and comparison
        const response = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219",
            max_tokens: 5000, 
            system: "This is a new conversation with ID: " + paperComparisonConversationId,
            thinking: {
                type: "enabled",
                budget_tokens: 3000
            },
            messages: [
                {
                    role: "user",
                    content: contentForPaperComparison
                }
            ] 
        });

        let result;
        if (response.content && response.content.length > 0) {
            // Find the text content block (not the thinking block)
            const textBlock = response.content.find(block => block.type === 'text');
            if (textBlock) {
                result = textBlock.text;
            } else {
                console.error('No text block found in response');
                result = 'No response content found';
            }
        } else {
            console.error('No content in response');
            result = 'No response received';
        }
        
        const outputFileName = `${zipName}_vs_${pdfName}_Run${runNumber}_CodeThenPaper.md`;
        const outputPath = path.join(outputDir, outputFileName);

        const mdContent = `# Codebase-Paper Consistency Analysis (Code-First Two-Stage)

**Code Archive:** ${zipName}
**Paper:** ${pdfName}  
**Analysis Date:** ${new Date().toISOString().split('T')[0]}

## Extracted Codebase Details (Stage 1 Output)
\`\`\`text
${extractedCodeDetails}
\`\`\`

## Paper Analysis and Comparison Results (Stage 2 Output)

${result}`;

        fs.writeFileSync(outputPath, mdContent);
        console.log(`Stage 2: Analysis saved to: ${outputPath}`);
        return result;
    } catch (error) {
        console.error('Error in Stage 2 (Paper Analysis and Comparison):', error);
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
        .option('-o, --output-dir <path>', 'Directory to save output files', './CS2 Results Output/CS2 CodeThenPaper results') // Configure this to your desired folder
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