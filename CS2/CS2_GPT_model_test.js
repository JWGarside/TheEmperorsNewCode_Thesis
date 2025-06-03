const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');
const { program } = require('commander');
const semanticChunking = require('./semanticChunking');

/**
 * Compare a research paper and code files/README from a zip archive using the OpenAI API.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} zipPath - Path to the zip archive containing code and README
 * @param {string} outputDir - Directory to save the output file
 * @param {number} runNumber - The run number for this comparison
 * @param {string} customPrompt - Custom prompt for OpenAI (uses default if null) 
 * @param {string} apiKeyCli - OpenAI API key from CLI (defaults to environment variable or hardcoded)
 * @returns {Promise<string>} - OpenAI's response
 */
async function comparePaperAndCode(pdfPath, zipPath, outputDir, runNumber, customPrompt = null, apiKeyCli = null) {
    // Initialize the OpenAI client
    const effectiveApiKey = process.env.OPENAI_API_KEY || "your-openai-key-here";
    if (!effectiveApiKey) {
        console.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable, provide --api-key, or ensure it's hardcoded.");
        throw new Error("OpenAI API key not found.");
    }
    const client = new OpenAI({ apiKey: effectiveApiKey });
    // Using gpt o4-mini as specified in the example
    const modelName = "o4-mini";

    console.log(`Reading PDF file "${pdfPath}" and encoding as base64...`);
    
    // Read and encode PDF file as Base64
    const pdfData = fs.readFileSync(pdfPath);
    const base64String = pdfData.toString("base64");

    // Semantically chunk the zip archive
    const { codeFiles, readmeContent } = await semanticChunking.chunkZipArchive(zipPath);

    // Extract file names for output naming
    const pdfName = path.basename(pdfPath, path.extname(pdfPath));
    const zipName = path.basename(zipPath, path.extname(zipPath));

    // Default prompt if none provided
    const analysisTaskPrompt = customPrompt || (
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

    console.log(`Analyzing paper "${pdfName}" with code from "${zipName}" using OpenAI API (Model: ${modelName})...`);

    try {
        // Construct the analysis text with code files and README
        let analysisText = "You are tasked with analyzing the provided research paper (PDF file) and its accompanying code for reproducibility based on the detailed analysis steps provided at the end.\n\n";
        
        analysisText += "--- CODE FILES ---\n";
        for (const file of codeFiles) {
            analysisText += `File: ${file.name}\nLanguage: ${file.language}\n\`\`\`${file.language || ''}\n${file.content}\n\`\`\`\n\n`;
        }
        
        analysisText += "--- README CONTENT ---\n";
        analysisText += (readmeContent || "No README content found.") + "\n\n";
        
        analysisText += "--- DETAILED ANALYSIS TASK ---\n";
        analysisText += analysisTaskPrompt;

        // Use OpenAI's recommended responses.create method with base64 PDF input
        const response = await client.responses.create({
            model: modelName, //configurable at top of the page
            max_output_tokens:20000, 
            reasoning: { 
                "effort": "high" //most equivalent option to 16000
            },
            input: [
                {
                    role: "user",
                    content: [
                        {
                            type: "input_file",
                            filename: path.basename(pdfPath),
                            file_data: `data:application/pdf;base64,${base64String}`,
                        },
                        {
                            type: "input_text",
                            text: analysisText,
                        },
                    ],
                },
            ],
        });
        
        let responseText = response.output_text || "";
        if (!responseText) {
            responseText = "No response content received from OpenAI.";
            console.warn("OpenAI response structure:", JSON.stringify(response, null, 2));
        }
        
        // Save result to markdown file  
        const outputFileName = `${zipName}_Run${runNumber}_OpenAI.md`;
        const outputPath = path.join(outputDir, outputFileName);

        // Create markdown content
        const mdContent = `# Paper-Code Consistency Analysis (OpenAI)

**Paper:** ${pdfName}  
**Code Archive:** ${zipName}
**Analysis Date:** ${new Date().toISOString().split('T')[0]}

## Analysis Results

${responseText}`;

        fs.writeFileSync(outputPath, mdContent);

        console.log(`Analysis saved to: ${outputPath}`);
        return responseText;
    } catch (error) {
        console.error('Error comparing paper and code with OpenAI:', error.message);
        if (error instanceof OpenAI.APIError) {
            console.error('OpenAI API Error Status:', error.status);
            console.error('OpenAI API Error Type:', error.type);
            console.error('OpenAI API Error Code:', error.code);
            console.error('OpenAI API Error Param:', error.param);
        } else {
            console.error('Error details:', error);
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
 * @param {string} customPrompt - Custom prompt for OpenAI (uses default if null)
 * @param {string} apiKey - OpenAI API key (defaults to environment variable or hardcoded)
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
        .name('paper-code-compare-openai')
        .description('Compare a research paper and code zip archive with OpenAI API in batch mode')
        .version('1.0.1')
        .requiredOption('-p, --pdf <path>', 'Path to the PDF research paper')
        .requiredOption('-z, --zip <path>', 'Path to the zip archive containing code and README') 
        .option('-o, --output-dir <path>', 'Directory to save output files', './CS2 Results output/CS2 o4 mini results') //Configure to your desired folder
        .option('-r, --runs <number>', 'Number of times to run each comparison', 3)
        .option('--prompt <text>', 'Custom prompt for OpenAI (uses default if not provided)')
        .option('--api-key <key>', 'OpenAI API key (optional, can use OPENAI_API_KEY environment variable or hardcoded key)')
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
                    parseInt(options.runs, 10),
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