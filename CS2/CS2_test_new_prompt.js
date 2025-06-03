// paper_code_compare_batch.js
const fs = require('fs');
const path = require('path');
const { Anthropic } = require('@anthropic-ai/sdk');
const { program } = require('commander');
const { v4: uuidv4 } = require('uuid');
const semanticChunking = require('./semanticChunking');

/**
 * Compare a research paper and code files/README from a zip archive using the Claude API.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} zipPath - Path to the zip archive containing code and README
 * @param {string} outputDir - Directory to save the output file
 * @param {number} runNumber - The run number for this comparison
 * @param {string} customPrompt - Custom prompt for Claude (uses default if null) 
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<string>} - Claude's response
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

    // Default prompt if none provided
    const prompt = customPrompt || (
        "# Research Code Reproducibility Analysis Prompt\n\n" +
         "Analyze the provided research paper (PDF) and code implementation (ZIP) to assess reproducibility:\n\n. " +
        "## Analysis Steps\n" +
        "1. Identify the paper's core claims and key methodological details\n. Note key methodological details explicitly described in the paper. Pay special attention to architecture specifications, algorithms, and parameter values. Identify which aspects are presented as fundamental to the approach versus optimization choices" +
        "2. Examine how core algorithms and architectures are implemented in the code\n. Trace the execution flow of any key components, noting any parameter, constants or design choices in the code. Think step-by-step when analysing the code" +
        "3. Note any discrepancies between paper descriptions and code implementation\n\n" +
        "## Discrepancy Classification\n" +
        "Classify discrepancies as:\n" +
        "- **Critical**: Prevent reproduction of core claims/methodology\n" +
        "- **Minor**: May affect performance but not fundamental approach\n" +
        "- **Cosmetic**: Documentation differences with minimal impact\n\n" +
        "## Output Format\n" +
        "1. Brief paper summary and core claims\n" +
        "2. Implementation assessment\n" +
        "3. Categorized discrepancies (if any)\n" +
        "4. Overall reproducibility conclusion\n\n" +
        "Remember that research code often differs from paper descriptions in minor ways. Focus on whether the implementation preserves the fundamental approach rather than perfect correspondence."
        );

    // Generate a unique conversation ID to prevent history issues
    const conversationId = uuidv4();

    console.log(`Analyzing paper "${pdfName}" with code from "${zipName}"`);

    try {
        // Construct the content for Claude API
        const content = [
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
                text: "I'm also providing code files extracted from a zip archive:\n\n"
            }
        ];

        // Add each code file
        for (const file of codeFiles) {
            content.push({
                type: "text",
                text: `File: ${file.name}\n\`\`\`${file.language}\n${file.content}\n\`\`\`\n`
            });
        }
        
        // Add the README content
        content.push({
            type: "text", 
            text: `README:\n${readmeContent}\n\n${prompt}`
        });

        // Send request to Claude API
        const response = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219",
            max_tokens: 1024,
            temperature: 0.5,
            system: "This is a new conversation with ID: " + conversationId,
            messages: [
                {
                    role: "user",
                    content: content
                }
            ] 
        });

        // Get Claude's response text
        const result = response.content[0].text;

        // Save result to markdown file  
        const outputFileName = `${zipName}_Run${runNumber}.md`;
        const outputPath = path.join(outputDir, outputFileName);

        // Create markdown content
        const mdContent = `# Paper-Code Consistency Analysis

**Paper:** ${pdfName}  
**Code Archive:** ${zipName}
**Analysis Date:** ${new Date().toISOString().split('T')[0]}

## Analysis Results

${result}`;

        fs.writeFileSync(outputPath, mdContent);

        console.log(`Analysis saved to: ${outputPath}`);
        return result;
    } catch (error) {
        console.error('Error comparing paper and code:', error);
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
        .option('-o, --output-dir <path>', 'Directory to save output files', './CS2 Results Output/CS2 new prompt results') // Configure this to your desired folder
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