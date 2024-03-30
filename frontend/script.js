let original_img_link = null
let initial_colour_id = null
let final_colour_id = null

document.getElementById('input_file').addEventListener('change', async function (event) {
    const uploadedImage = event.target.files[0];
    
    const image = document.getElementById('graph');
    original_img_link = URL.createObjectURL(uploadedImage);
    image.src = original_img_link;
    
    const target_url = "http://127.0.0.1:8000/process_image";
    try {
        const formData = new FormData();
        formData.append("file", uploadedImage);
    
        const response = await fetch(target_url, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        // Extract colors from the fetched data
        const colors = [];
        for (let i = 0; i < data.buttons.length; i++) {
            const color = data.buttons[i].color;
            colors.push(color);
        }
                
        // Call function to create buttons with the extracted colors
        document.getElementById("instruction_change_from").hidden = false;
        document.getElementById("instruction_change_to").hidden = false;
        
        createButtons(colors);
    } catch (error) {
        console.error(error);
    }
});

function createButtons(colors) {
    const buttonContainer = document.getElementById('button-container');
    buttonContainer.innerHTML = ""; // Clear previous buttons (if any)
    let idx = 1;
    colors.forEach(color => {
        const button = document.createElement('button');
        button.id = idx.toString();
        idx++;
        
        button.style.backgroundColor = color;
        button.className = "palette-button";
        button.addEventListener('click', select_initial_colour);
        
        buttonContainer.appendChild(button);
    });
}

// Function to handle button click
function select_initial_colour(event) {
    const clickedButton = event.target;
    const buttonContainer = document.getElementById('button-container');
    const buttons = buttonContainer.querySelectorAll('.palette-button');
    initial_colour_id = clickedButton.id
    buttons.forEach(button => {
        // Do something with each button
        button.style.border = "2px solid white";
    });

    clickedButton.style.border = "2px solid black";
}

document.getElementById('reset').addEventListener('click', async function () {
    var image = document.getElementById('graph');
    image.src = original_img_link;
    const url = 'http://127.0.0.1:8000/reset'
    await fetch(url, {
        method: "GET"
    });
});

const colorPicker = document.querySelector("#color-picker");
colorPicker.addEventListener("change", function (event) {
    final_colour_id = event.target.value;
}, false);

document.getElementById("apply").addEventListener('click', async function(event) {
    const formData = new FormData();
    formData.append("change", initial_colour_id);
    formData.append("new_colour_hex", final_colour_id);
    
    const response = await fetch('http://127.0.0.1:8000/change/', {
        method: 'POST',
        body: formData
    });
    const blob = await response.blob();
    const objectURL = URL.createObjectURL(blob);

    const image = document.getElementById("graph");
    image.src = objectURL;

    const button = document.getElementById(initial_colour_id);
    button.style.backgroundColor = final_colour_id;
});
