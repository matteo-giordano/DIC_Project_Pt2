/**
 * Dataset Generator Configuration Manager
 * Handles configuration state persistence across generator tabs
 */

// Key for storing configuration in local storage
const CONFIG_STORAGE_KEY = 'dataset_generator_config';

// Function to save current configuration to localStorage
function saveConfigurationState() {
    // Create configuration object
    const config = {
        datasetName: document.getElementById('dataset-name')?.textContent || '',
        generators: {},
        globalSettings: {
            ensureConnected: document.getElementById('ensure-connected-global')?.checked || true,
            ensureUnique: document.getElementById('ensure-unique-global')?.checked || true,
            countPerConfig: document.getElementById('count-per-config')?.value || 10
        }
    };
    
    // Collect generator configuration
    document.querySelectorAll('.config-section[data-generator]').forEach(section => {
        const generatorType = section.getAttribute('data-generator');
        const isEnabled = section.getAttribute('data-enabled') === 'true';
        
        // Basic generator info
        config.generators[generatorType] = {
            enabled: isEnabled
        };
        
        // Skip collecting detailed params if generator is disabled
        if (!isEnabled) return;
        
        // Maps per configuration
        const countInput = section.querySelector(`input[name="${generatorType}-count_per_config"]`);
        if (countInput) {
            config.generators[generatorType].countPerConfig = parseInt(countInput.value) || 10;
        }
        
        // Width values
        const widthValues = [];
        section.querySelectorAll(`input[name="${generatorType}-width[]"]`).forEach(input => {
            widthValues.push(parseInt(input.value) || 21);
        });
        config.generators[generatorType].width = widthValues;
        
        // Height values
        const heightValues = [];
        section.querySelectorAll(`input[name="${generatorType}-height[]"]`).forEach(input => {
            heightValues.push(parseInt(input.value) || 21);
        });
        config.generators[generatorType].height = heightValues;
        
        // Generator-specific parameters
        if (generatorType === 'terrain') {
            const blobDensityValues = [];
            section.querySelectorAll(`input[name="${generatorType}-blob_density[]"]`).forEach(input => {
                blobDensityValues.push(parseFloat(input.value) || 0.005);
            });
            config.generators[generatorType].blobDensity = blobDensityValues;
        } else if (generatorType === 'gaussian') {
            // Threshold values
            const thresholdValues = [];
            section.querySelectorAll(`input[name="${generatorType}-threshold[]"]`).forEach(input => {
                thresholdValues.push(parseFloat(input.value) || 0.65);
            });
            config.generators[generatorType].threshold = thresholdValues;
            
            // Sigma X values
            const sigmaXValues = [];
            section.querySelectorAll(`input[name="${generatorType}-sigma_x[]"]`).forEach(input => {
                sigmaXValues.push(parseFloat(input.value) || 1.0);
            });
            config.generators[generatorType].sigmaX = sigmaXValues;
            
            // Sigma Y values
            const sigmaYValues = [];
            section.querySelectorAll(`input[name="${generatorType}-sigma_y[]"]`).forEach(input => {
                sigmaYValues.push(parseFloat(input.value) || 1.0);
            });
            config.generators[generatorType].sigmaY = sigmaYValues;
        }
    });
    
    // Save to localStorage
    try {
        localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
        console.log('Configuration saved to localStorage');
    } catch (error) {
        console.error('Failed to save configuration to localStorage:', error);
    }
}

// Function to load configuration state from localStorage
function loadConfigurationState() {
    try {
        // Get saved configuration
        const savedConfig = localStorage.getItem(CONFIG_STORAGE_KEY);
        if (!savedConfig) {
            console.log('No saved configuration found');
            return false;
        }
        
        const config = JSON.parse(savedConfig);
        console.log('Loaded configuration:', config);
        
        // Apply dataset name
        if (config.datasetName && document.getElementById('dataset-name')) {
            document.getElementById('dataset-name').textContent = config.datasetName;
        }
        
        // Apply global settings
        if (config.globalSettings) {
            const { ensureConnected, ensureUnique, countPerConfig } = config.globalSettings;
            
            // Set ensure connected checkbox
            const ensureConnectedGlobal = document.getElementById('ensure-connected-global');
            if (ensureConnectedGlobal) {
                ensureConnectedGlobal.checked = ensureConnected;
                
                // Also set the individual checkbox if available
                const ensureConnectedIndividual = document.getElementById('ensure-connected');
                if (ensureConnectedIndividual) {
                    ensureConnectedIndividual.checked = ensureConnected;
                }
            }
            
            // Set ensure unique checkbox
            const ensureUniqueGlobal = document.getElementById('ensure-unique-global');
            if (ensureUniqueGlobal) {
                ensureUniqueGlobal.checked = ensureUnique;
                
                // Also set the individual checkbox if available
                const ensureUniqueIndividual = document.getElementById('ensure-unique');
                if (ensureUniqueIndividual) {
                    ensureUniqueIndividual.checked = ensureUnique;
                }
            }
            
            // Set count per config
            const countPerConfigInput = document.getElementById('count-per-config');
            if (countPerConfigInput) {
                countPerConfigInput.value = countPerConfig;
            }
        }
        
        // Apply generator configurations
        if (config.generators) {
            document.querySelectorAll('.config-section[data-generator]').forEach(section => {
                const generatorType = section.getAttribute('data-generator');
                const generatorConfig = config.generators[generatorType];
                
                if (!generatorConfig) return;
                
                // Set enabled state
                const toggleCheckbox = section.querySelector('.generator-toggle');
                if (toggleCheckbox) {
                    toggleCheckbox.checked = generatorConfig.enabled;
                    section.setAttribute('data-enabled', generatorConfig.enabled.toString());
                    
                    // Show/hide content based on enabled state
                    const content = section.querySelector('.config-content');
                    if (content) {
                        content.style.display = generatorConfig.enabled ? 'block' : 'none';
                    }
                }
                
                // Skip applying detailed params if generator is disabled
                if (!generatorConfig.enabled) return;
                
                // Apply count per config
                const countInput = section.querySelector(`input[name="${generatorType}-count_per_config"]`);
                if (countInput && generatorConfig.countPerConfig) {
                    countInput.value = generatorConfig.countPerConfig;
                }
                
                // Apply width values
                if (generatorConfig.width && generatorConfig.width.length > 0) {
                    const widthContainer = section.querySelector(`.array-container[data-param-name="${generatorType}-width"]`);
                    if (widthContainer) {
                        // Clear existing inputs
                        widthContainer.innerHTML = '';
                        
                        // Add new inputs for each width
                        generatorConfig.width.forEach(width => {
                            const arrayItem = document.createElement('div');
                            arrayItem.className = 'array-item';
                            
                            const input = document.createElement('input');
                            input.type = 'number';
                            input.className = 'array-item-value';
                            input.name = `${generatorType}-width[]`;
                            input.value = width;
                            input.min = '3';
                            input.max = '101';
                            input.step = '2';
                            
                            const removeBtn = document.createElement('button');
                            removeBtn.type = 'button';
                            removeBtn.className = 'array-item-remove';
                            removeBtn.textContent = '×';
                            
                            arrayItem.appendChild(input);
                            arrayItem.appendChild(removeBtn);
                            widthContainer.appendChild(arrayItem);
                        });
                    }
                }
                
                // Apply height values
                if (generatorConfig.height && generatorConfig.height.length > 0) {
                    const heightContainer = section.querySelector(`.array-container[data-param-name="${generatorType}-height"]`);
                    if (heightContainer) {
                        // Clear existing inputs
                        heightContainer.innerHTML = '';
                        
                        // Add new inputs for each height
                        generatorConfig.height.forEach(height => {
                            const arrayItem = document.createElement('div');
                            arrayItem.className = 'array-item';
                            
                            const input = document.createElement('input');
                            input.type = 'number';
                            input.className = 'array-item-value';
                            input.name = `${generatorType}-height[]`;
                            input.value = height;
                            input.min = '3';
                            input.max = '101';
                            input.step = '2';
                            
                            const removeBtn = document.createElement('button');
                            removeBtn.type = 'button';
                            removeBtn.className = 'array-item-remove';
                            removeBtn.textContent = '×';
                            
                            arrayItem.appendChild(input);
                            arrayItem.appendChild(removeBtn);
                            heightContainer.appendChild(arrayItem);
                        });
                    }
                }
                
                // Apply generator-specific parameters
                if (generatorType === 'terrain' && generatorConfig.blobDensity && generatorConfig.blobDensity.length > 0) {
                    const densityContainer = section.querySelector(`.array-container[data-param-name="${generatorType}-blob_density"]`);
                    if (densityContainer) {
                        // Clear existing inputs
                        densityContainer.innerHTML = '';
                        
                        // Add new inputs for each blob density
                        generatorConfig.blobDensity.forEach(density => {
                            const arrayItem = document.createElement('div');
                            arrayItem.className = 'array-item';
                            
                            const input = document.createElement('input');
                            input.type = 'number';
                            input.className = 'array-item-value';
                            input.name = `${generatorType}-blob_density[]`;
                            input.value = density;
                            input.min = '0.001';
                            input.max = '0.05';
                            input.step = '0.001';
                            
                            const removeBtn = document.createElement('button');
                            removeBtn.type = 'button';
                            removeBtn.className = 'array-item-remove';
                            removeBtn.textContent = '×';
                            
                            arrayItem.appendChild(input);
                            arrayItem.appendChild(removeBtn);
                            densityContainer.appendChild(arrayItem);
                        });
                    }
                } else if (generatorType === 'gaussian') {
                    // Apply threshold values
                    if (generatorConfig.threshold && generatorConfig.threshold.length > 0) {
                        const thresholdContainer = section.querySelector(`.array-container[data-param-name="${generatorType}-threshold"]`);
                        if (thresholdContainer) {
                            // Clear existing inputs
                            thresholdContainer.innerHTML = '';
                            
                            // Add new inputs for each threshold
                            generatorConfig.threshold.forEach(threshold => {
                                const arrayItem = document.createElement('div');
                                arrayItem.className = 'array-item';
                                
                                const input = document.createElement('input');
                                input.type = 'number';
                                input.className = 'array-item-value';
                                input.name = `${generatorType}-threshold[]`;
                                input.value = threshold;
                                input.min = '0.5';
                                input.max = '0.9';
                                input.step = '0.05';
                                
                                const removeBtn = document.createElement('button');
                                removeBtn.type = 'button';
                                removeBtn.className = 'array-item-remove';
                                removeBtn.textContent = '×';
                                
                                arrayItem.appendChild(input);
                                arrayItem.appendChild(removeBtn);
                                thresholdContainer.appendChild(arrayItem);
                            });
                        }
                    }
                    
                    // Apply sigma X values
                    if (generatorConfig.sigmaX && generatorConfig.sigmaX.length > 0) {
                        const sigmaXContainer = section.querySelector(`.array-container[data-param-name="${generatorType}-sigma_x"]`);
                        if (sigmaXContainer) {
                            // Clear existing inputs
                            sigmaXContainer.innerHTML = '';
                            
                            // Add new inputs for each sigma X
                            generatorConfig.sigmaX.forEach(sigma => {
                                const arrayItem = document.createElement('div');
                                arrayItem.className = 'array-item';
                                
                                const input = document.createElement('input');
                                input.type = 'number';
                                input.className = 'array-item-value';
                                input.name = `${generatorType}-sigma_x[]`;
                                input.value = sigma;
                                input.min = '0.1';
                                input.max = '5.0';
                                input.step = '0.1';
                                
                                const removeBtn = document.createElement('button');
                                removeBtn.type = 'button';
                                removeBtn.className = 'array-item-remove';
                                removeBtn.textContent = '×';
                                
                                arrayItem.appendChild(input);
                                arrayItem.appendChild(removeBtn);
                                sigmaXContainer.appendChild(arrayItem);
                            });
                        }
                    }
                    
                    // Apply sigma Y values
                    if (generatorConfig.sigmaY && generatorConfig.sigmaY.length > 0) {
                        const sigmaYContainer = section.querySelector(`.array-container[data-param-name="${generatorType}-sigma_y"]`);
                        if (sigmaYContainer) {
                            // Clear existing inputs
                            sigmaYContainer.innerHTML = '';
                            
                            // Add new inputs for each sigma Y
                            generatorConfig.sigmaY.forEach(sigma => {
                                const arrayItem = document.createElement('div');
                                arrayItem.className = 'array-item';
                                
                                const input = document.createElement('input');
                                input.type = 'number';
                                input.className = 'array-item-value';
                                input.name = `${generatorType}-sigma_y[]`;
                                input.value = sigma;
                                input.min = '0.1';
                                input.max = '5.0';
                                input.step = '0.1';
                                
                                const removeBtn = document.createElement('button');
                                removeBtn.type = 'button';
                                removeBtn.className = 'array-item-remove';
                                removeBtn.textContent = '×';
                                
                                arrayItem.appendChild(input);
                                arrayItem.appendChild(removeBtn);
                                sigmaYContainer.appendChild(arrayItem);
                            });
                        }
                    }
                }
            });
        }
        
        return true;
    } catch (error) {
        console.error('Failed to load configuration from localStorage:', error);
        return false;
    }
}

// Function to attach event listeners for inputs to save configuration state
function setupConfigChangeListeners() {
    // Listen for changes on checkboxes, inputs, and generator toggles
    document.querySelectorAll('input, select').forEach(input => {
        input.addEventListener('change', saveConfigurationState);
    });
    
    // Listen for array item additions and removals
    document.querySelectorAll('.array-add').forEach(btn => {
        btn.addEventListener('click', () => {
            // Wait a moment for the DOM to update
            setTimeout(saveConfigurationState, 50);
        });
    });
    
    // Set up a mutation observer to listen for dynamically added/removed array items
    const arrayContainers = document.querySelectorAll('.array-container');
    if (arrayContainers.length > 0) {
        const observer = new MutationObserver(() => {
            saveConfigurationState();
        });
        
        arrayContainers.forEach(container => {
            observer.observe(container, { childList: true, subtree: true });
        });
    }
    
    // Listen for dataset name changes
    document.getElementById('dataset-name')?.addEventListener('DOMSubtreeModified', saveConfigurationState);
}

// Initialize configuration persistence
document.addEventListener('DOMContentLoaded', function() {
    // First try to load saved configuration
    const configLoaded = loadConfigurationState();
    
    // Set up change listeners
    setupConfigChangeListeners();
    
    console.log('Dataset configurator initialized' + (configLoaded ? ' with saved configuration' : ''));
});

// Save configuration when leaving the page
window.addEventListener('beforeunload', saveConfigurationState);

// Generator Configuration UI Logic

// Initialize configuration UI
function initGeneratorConfig() {
    // Setup toggle switches for each generator
    document.querySelectorAll('.config-header').forEach(header => {
        header.addEventListener('click', function(e) {
            // Don't toggle if clicking on the checkbox itself
            if (e.target.type !== 'checkbox') {
                const checkbox = this.querySelector('input[type="checkbox"]');
                checkbox.checked = !checkbox.checked;
                toggleGeneratorSection(checkbox);
            }
        });
    });

    // Setup checkbox handlers
    document.querySelectorAll('.generator-toggle').forEach(toggle => {
        toggle.addEventListener('change', function() {
            toggleGeneratorSection(this);
        });
        
        // Initialize sections based on initial checkbox state
        toggleGeneratorSection(toggle);
    });
    
    // Make sure all config content sections are properly initialized
    document.querySelectorAll('.config-content').forEach(content => {
        if (!content.closest('.config-section').querySelector('.generator-toggle').checked) {
            content.style.display = 'none';
        } else {
            content.style.display = 'block';
        }
    });
    
    // Add event handlers for array items
    setupArrayControls();
    
    // Setup save button
    document.getElementById('save-generator-config').addEventListener('click', saveGeneratorConfig);
    
    // Check if panel visibility state is stored in localStorage
    loadPanelState();
}

// Toggle generator configuration section visibility
function toggleGeneratorSection(checkbox) {
    const section = checkbox.closest('.config-section');
    const content = section.querySelector('.config-content');
    
    if (checkbox.checked) {
        content.style.display = 'block';
        content.classList.add('active');
        section.classList.add('active');
    } else {
        content.style.display = 'none';
        content.classList.remove('active');
        section.classList.remove('active');
    }
}

// Save the panel visibility state to localStorage
function savePanelState(isVisible) {
    localStorage.setItem('configPanelVisible', isVisible ? 'true' : 'false');
}

// Load the panel visibility state from localStorage
function loadPanelState() {
    const configPanel = document.getElementById('generator-config-panel');
    const columnsContainer = document.querySelector('.content-columns');
    
    if (!configPanel || !columnsContainer) return;
    
    const savedState = localStorage.getItem('configPanelVisible');
    
    // If we have a saved state and it's false (hidden)
    if (savedState === 'false') {
        configPanel.classList.add('hidden');
        columnsContainer.classList.add('single-column');
    }
}

// Toggle the panel visibility and save the state
function toggleConfigPanel() {
    const configPanel = document.getElementById('generator-config-panel');
    const columnsContainer = document.querySelector('.content-columns');
    
    if (!configPanel || !columnsContainer) return;
    
    if (configPanel.classList.contains('hidden')) {
        // Show config panel
        configPanel.classList.remove('hidden');
        columnsContainer.classList.remove('single-column');
        savePanelState(true);
    } else {
        // Hide config panel
        configPanel.classList.add('hidden');
        columnsContainer.classList.add('single-column');
        savePanelState(false);
    }
}

// Setup controls for array parameters
function setupArrayControls() {
    // Add new array item
    document.querySelectorAll('.array-add').forEach(btn => {
        btn.addEventListener('click', function() {
            const container = this.previousElementSibling;
            const newItem = document.createElement('div');
            newItem.className = 'array-item';
            
            const input = document.createElement('input');
            const paramType = container.dataset.paramType || 'text';
            input.type = paramType === 'number' ? 'number' : 'text';
            
            // If it's a number input, add any necessary attributes
            if (input.type === 'number') {
                if (container.dataset.paramMin) {
                    input.min = container.dataset.paramMin;
                }
                if (container.dataset.paramMax) {
                    input.max = container.dataset.paramMax;
                }
                if (container.dataset.paramStep) {
                    input.step = container.dataset.paramStep;
                }
            }
            
            input.className = 'array-item-value';
            input.name = container.dataset.paramName + '[]';
            
            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'array-item-remove';
            removeBtn.textContent = '×';
            removeBtn.addEventListener('click', function() {
                newItem.remove();
            });
            
            newItem.appendChild(input);
            newItem.appendChild(removeBtn);
            container.appendChild(newItem);
        });
    });
    
    // Remove array item
    document.querySelectorAll('.array-item-remove').forEach(btn => {
        btn.addEventListener('click', function() {
            this.parentElement.remove();
        });
    });
}

// Save generator configuration and generate dataset
function saveGeneratorConfig() {
    const saveBtn = document.getElementById('save-generator-config');
    saveBtn.textContent = 'Saving Config...';
    saveBtn.disabled = true;
    
    // Build configuration object
    const config = {
        name: document.getElementById('dataset-name').textContent.trim(),
        generators: {}
    };
    
    // Process each generator
    document.querySelectorAll('.config-section').forEach(section => {
        const generatorType = section.dataset.generatorType;
        const isEnabled = section.querySelector('.generator-toggle').checked;
        
        if (isEnabled) {
            config.generators[generatorType] = {};
            
            // Get count per config
            const countInput = section.querySelector('.count-per-config');
            config.generators[generatorType]['count_per_config'] = parseInt(countInput.value) || 10;
            
            // Get single parameters
            section.querySelectorAll('.param-value:not([data-is-array="true"])').forEach(input => {
                if (input.name !== 'count_per_config') {
                    const paramName = input.name.replace(`${generatorType}-`, '');
                    // Convert to appropriate type
                    if (input.type === 'number') {
                        if (input.step === '1' || input.step === '1.0') {
                            config.generators[generatorType][paramName] = parseInt(input.value);
                        } else {
                            config.generators[generatorType][paramName] = parseFloat(input.value);
                        }
                    } else {
                        config.generators[generatorType][paramName] = input.value;
                    }
                }
            });
            
            // Get array parameters
            section.querySelectorAll('.array-container').forEach(container => {
                const paramName = container.dataset.paramName.replace(`${generatorType}-`, '');
                const values = [];
                
                container.querySelectorAll('.array-item-value').forEach(input => {
                    // Convert to appropriate type
                    if (container.dataset.paramType === 'number') {
                        if (container.dataset.paramStep === '1' || container.dataset.paramStep === '1.0') {
                            values.push(parseInt(input.value));
                        } else {
                            values.push(parseFloat(input.value));
                        }
                    } else {
                        values.push(input.value);
                    }
                });
                
                if (values.length > 0) {
                    config.generators[generatorType][paramName] = values;
                }
            });
        }
    });
    
    // Convert to YAML and send to server
    const yamlData = convertToYAML(config);
    
    // Show progress container
    const progressContainer = document.getElementById('generate-progress-container');
    const progressBar = document.getElementById('generate-progress-bar');
    const progressStatus = document.getElementById('generate-progress-status');
    
    progressContainer.style.display = 'block';
    progressBar.style.width = '10%';
    progressStatus.textContent = 'Saving configuration...';
    
    fetch('/save_config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ config: yamlData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            progressBar.style.width = '20%';
            progressStatus.textContent = 'Configuration saved, starting dataset generation...';
            
            // Start generating the dataset
            setTimeout(() => {
                generateFullDataset(progressBar, progressStatus);
            }, 500);
        } else {
            alert('Error: ' + data.error);
            saveBtn.textContent = 'Generate Dataset';
            saveBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
    })
    .catch(error => {
        alert('Error: ' + error.message);
        saveBtn.textContent = 'Generate Dataset';
        saveBtn.disabled = false;
        progressContainer.style.display = 'none';
    });
}

// Generate the full dataset
function generateFullDataset(progressBar, progressStatus) {
    // Update progress
    progressBar.style.width = '20%';
    progressStatus.textContent = 'Initializing dataset generator...';
    
    // Call the generate_full_dataset endpoint
    fetch('/generate_full_dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Simulating progress updates, as the actual process might be too fast for a meaningful loading bar
            progressBar.style.width = '40%';
            progressStatus.textContent = 'Generating maze maps...';
            
            setTimeout(() => {
                progressBar.style.width = '70%';
                progressStatus.textContent = `Created ${data.map_count} maze maps...`;
                
                setTimeout(() => {
                    progressBar.style.width = '90%';
                    progressStatus.textContent = 'Saving dataset files...';
                    
                    setTimeout(() => {
                        progressBar.style.width = '100%';
                        progressStatus.textContent = `Dataset "${data.dataset_name}" generated successfully!`;
                        
                        // After a pause, redirect to the dataset visualizer
                        setTimeout(() => {
                            // Save the dataset path to localStorage for the visualizer to use
                            localStorage.setItem('lastDatasetPath', data.dataset_path);
                            
                            // Create a form to redirect to the dataset visualizer
                            const form = document.createElement('form');
                            form.method = 'POST';
                            form.action = '/load_dataset';
                            form.style.display = 'none';
                            
                            const input = document.createElement('input');
                            input.type = 'hidden';
                            input.name = 'dataset_path';
                            input.value = data.dataset_path;
                            
                            form.appendChild(input);
                            document.body.appendChild(form);
                            form.submit();
                        }, 1500);
                    }, 500);
                }, 800);
            }, 1000);
        } else {
            // Show error
            progressBar.style.width = '0%';
            progressStatus.textContent = 'Error: ' + data.error;
            progressStatus.style.color = 'var(--error-color)';
            
            // Re-enable the button after a delay
            setTimeout(() => {
                document.getElementById('save-generator-config').textContent = 'Generate Dataset';
                document.getElementById('save-generator-config').disabled = false;
            }, 2000);
        }
    })
    .catch(error => {
        // Show error
        progressBar.style.width = '0%';
        progressStatus.textContent = 'Error: ' + error.message;
        progressStatus.style.color = 'var(--error-color)';
        
        // Re-enable the button after a delay
        setTimeout(() => {
            document.getElementById('save-generator-config').textContent = 'Generate Dataset';
            document.getElementById('save-generator-config').disabled = false;
        }, 2000);
    });
}

// Simple function to convert JS object to YAML string
function convertToYAML(obj) {
    let yaml = `name: "${obj.name}"\n\ngenerators:\n`;
    
    for (const [genType, genConfig] of Object.entries(obj.generators)) {
        yaml += `  ${genType}:\n`;
        
        for (const [key, value] of Object.entries(genConfig)) {
            if (Array.isArray(value)) {
                // Format array values properly
                const formattedValues = value.map(v => {
                    // If it's a number, don't add quotes
                    if (typeof v === 'number') {
                        return v;
                    } else {
                        // Add quotes for strings
                        return `"${v}"`;
                    }
                }).join(', ');
                
                yaml += `    ${key}: [${formattedValues}]\n`;
            } else {
                // Format scalar values
                if (typeof value === 'string') {
                    yaml += `    ${key}: "${value}"\n`;
                } else {
                    yaml += `    ${key}: ${value}\n`;
                }
            }
        }
    }
    
    return yaml;
}

// Initialize when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initGeneratorConfig); 