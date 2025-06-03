// paper_code_compare_batch.js
const fs = require('fs');
const path = require('path');
const { Anthropic } = require('@anthropic-ai/sdk');
const { program } = require('commander');
const { v4: uuidv4 } = require('uuid');

/**
 * Compare a research paper and code file using the Claude API to identify discrepancies.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} codePath - Path to the code file
 * @param {string} outputDir - Directory to save the output file
 * @param {number} runNumber - The run number for this comparison
 * @param {string} customPrompt - Custom prompt for Claude (uses default if null)
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<string>} - Claude's response
 */
async function comparePaperAndCode(pdfPath, codePath, outputDir, runNumber, customPrompt = null, apiKey = null) {
    // Initialize the Anthropic client
    const anthropic = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY || "your-anthropic-key-here"
    });
    
    // Read and encode PDF file as Base64
    const pdfData = fs.readFileSync(pdfPath).toString('base64');
    
    // Read the code file
    const codeContent = fs.readFileSync(codePath, 'utf-8');
    
    // Determine file extension to help Claude understand the language
    const fileExtension = path.extname(codePath).slice(1);
    
    // Extract file names for output naming
    const pdfName = path.basename(pdfPath, path.extname(pdfPath));
    const codeName = path.basename(codePath, path.extname(codePath));
    
    // Default prompt if none provided
    const prompt = customPrompt || (
        "Compare the provided research paper and code implementation to identify any discrepancies that could affect the reproducibility or validity of the work. Focus on differences in methodology, algorithms, mathematical approaches, or key implementation details that might lead to different results or impede replication. Ignore minor deviations in code style, variable names, or superficial details if they are unlikely to change the interpretation or reproducibility of the work. After analyzing both artifacts, summarize your findings: For each identified discrepancy, provide a description, references to the relevant paper section(s) and code location(s), and an explanation of how it could affect reproducibility or validity. If no discrepancies are found that have a realistic chance of impacting the reproducibility or validity of the work, simply state that the code implementation accurately represents the paper's described methodology. If no discrepancies are found, begin your response with \"NO DISCREPANCIES FOUND\""
    );
    
    // Generate a unique conversation ID to prevent history issues
    const conversationId = uuidv4();
    
    console.log(`Analyzing paper "${pdfName}" with code "${codeName}"`);
    
    try {
        // Send request to Claude API
        const response = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219",
            max_tokens: 1024,
            temperature: 0.5,
            system: "This is a new conversation with ID: " + conversationId,
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
                            text: `I'm also providing a ${fileExtension} code file:\n\n\`\`\`${fileExtension}\n${codeContent}\n\`\`\`\n\n${prompt}`
                        }
                    ]
                }
            ]
        });
        
        // Get Claude's response text
        const result = response.content[0].text;
        
        // Save result to markdown file
        const outputFileName = `${codeName}_Run${runNumber}.md`;
        const outputPath = path.join(outputDir, outputFileName);
        
        // Create markdown content
        const mdContent = `# Paper-Code Consistency Analysis

**Paper:** ${pdfName}
**Code:** ${codeName}
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
 * Run a batch of comparisons between a paper and code file multiple times.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} codePath - Path to the code file
 * @param {string} outputDir - Directory to save the output file
 * @param {number} numRuns - Number of times to run each comparison (default: 3)
 * @param {string} customPrompt - Custom prompt for Claude (uses default if null)
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<void>}
 */
async function runComparisonsInBatch(pdfPath, codePath, outputDir, numRuns = 3, customPrompt = null, apiKey = null) {
    for (let i = 1; i <= numRuns; i++) {
        console.log(`Running comparison ${i} of ${numRuns}...`);
        await comparePaperAndCode(pdfPath, codePath, outputDir, i, customPrompt, apiKey);
    }
}

// Command line interface setup
if (require.main === module) {
    program
        .name('paper-code-compare-batch')
        .description('Compare a research paper and code with Claude AI in batch mode')
        .version('1.0.0')
        .requiredOption('-p, --pdf <path>', 'Path to the PDF research paper')
        .requiredOption('-c, --code <path>', 'Path to the code file to analyze')
        .option('-o, --output-dir <path>', 'Directory to save output files', './Results')
        .option('-r, --runs <number>', 'Number of times to run each comparison', 3)
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
                    options.code,
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