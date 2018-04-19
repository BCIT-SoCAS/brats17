var elem = document.querySelector('select');
var instance = M.FormSelect.init(elem);

// Or with jQuery

$(document).ready(function(){
    $('select').formSelect();
});


var mri_images
var scan_types = {
    flair: 0,
    t1: 1,
    t1c: 2,
    t2: 3
}
current_scan = 'flair'

let view = {
    ax: {
        view: 'top',
        brain: document.getElementById('ax_b'),
        tumor: document.getElementById('ax_t'),
        range: document.getElementById('ax_range'),
        index: document.getElementById('ax_index')
    },
    sg: {
        view: 'side',
        brain: document.getElementById('sg_b'),
        tumor: document.getElementById('sg_t'),
        range: document.getElementById('sg_range'),
        index: document.getElementById('sg_index')
    },
    cr: {
        view: 'front',
        brain: document.getElementById('cr_b'),
        tumor: document.getElementById('cr_t'),
        range: document.getElementById('cr_range'),
        index: document.getElementById('cr_index')
    }
}

function initView() {

    var scan_type = document.getElementById('scan_type');
    var download = document.getElementById('download');

    scan_type.selectedIndex = scan_types[current_scan]

    for (var key in view) {
        view[key].brain.draw = draw;
        view[key].tumor.draw = draw;
    }

    scan_type.addEventListener('change', function(event) {
        mri_images.current.data = mri_images.scans[this.value];
        drawImages();
    })
}

function main() {
    
    initView();
    
    //request MRIs
    postRequest('/getMRIs').then(data=>data.arrayBuffer())
        .then(data=>msgpack.decode(data))
        .then(data=>{
            initMRIImages(data);
            drawImages();

            //remove the overlay
            overlay = document.getElementById("overlay");
            overlay.style.visibility = 'hidden'

            //request for the prediction
            console.log('predicting')
            postRequest('/predict').then(response => response.json())
                .then(data => {
                    console.log(data)

                    let prediction_div = document.getElementById("prediction");
                    //delete everything else
                    prediction_div.innerHTML = ""
                    prediction_div.appendChild(create_pred_div(data));
                    
                })

        })

    // TODO uncomment this
    

}

function create_pred_div(data) {
    
    var result_div = document.createElement('div');

    let labels = ['Less than 250 Days','250 to 500 Days','More than 500 Days']

    let title_div = document.createElement('h5');
    title_div.innerHTML = 'Survivability Prediction';
    title_div.style.color = 'white';
    result_div.appendChild(title_div);
    
    for(let x = 0 ; x < labels.length ; x++) {

        let prediction_div = document.createElement('div');
        prediction_div.style.color = 'white';
        prediction_div.style.fontWeight = 'bold';
        prediction_div.innerHTML = `${labels[x]}: ${data[labels[x]]}%`
        result_div.appendChild(prediction_div); 
    }
    
    let note_div = document.createElement('div')
    note_div.style.color = 'white';
    note_div.style.fontStyle = 'italic'
    note_div.innerHTML = '*Note: These predictions are ~50% accurate'
    result_div.appendChild(note_div);

    return result_div;
}

function initMRIImages(data) {


    mri_images = {
        scans: data.brain,
        current: {
            data: data.brain[current_scan],
            getSlice: getSlice,
            dim: {
                depth: data.brain[current_scan].length,
                height:data.brain[current_scan][0].length,
                width: data.brain[current_scan][0][0].length
            }
        },
        tumor: {
            data: data.tumor,
            getSlice: getSlice,
            dim: { 
                depth: data.tumor.length,
                height:data.tumor[0].length,
                width: data.tumor[0][0].length
            }
        }
    }

    // initialize index values and set default  
    half_depth = Math.floor(data.brain[current_scan].length / 2);
    half_width = Math.floor(data.brain[current_scan][0][0].length / 2);
    half_height = Math.floor(data.brain[current_scan][0].length / 2);
    
    view.ax.range.max = mri_images.current.dim.depth;
    view.sg.range.max = mri_images.current.dim.width;
    view.cr.range.max = mri_images.current.dim.height

    view.ax.range.value = half_depth;
    view.sg.range.value = half_width;
    view.cr.range.value = half_height;

    view.ax.range.value = half_depth;
    view.sg.range.value = half_width;
    view.cr.range.value = half_height;

    for (let key in view){
        let v = view[key]

        v.index.value = v.range.value; 
        v.range.addEventListener('change', function(event) {
            v.index.value = v.range.value;
            v.brain.draw(mri_images.current.getSlice(v.view,v.range.value), 
                'grayscale', v.view==='side' || v.view==='front', v.view==='side')
            v.tumor.draw(mri_images.tumor.getSlice(v.view,v.range.value), 
                'grayscale', v.view==='side' || v.view==='front', v.view==='side') 
        })
    }
    
} 

function drawImages() {
    for (var key in view) {
        let v = view[key]
        
        index = v.range.value
        v.brain.draw(mri_images.current.getSlice(v.view,index),'grayscale',
            v.view==='side' || v.view==='front', v.view==='side');
        v.tumor.draw(mri_images.tumor.getSlice(v.view,index),'grayscale',
            v.view==='side' || v.view==='front', v.view==='side');
    }
}

function getSlice(type="top", index) {

    if(type === 'top') {
        if(index >= this.data.length) {
            index = this.data.length - 1;
        }

        if(index < 0) {
            index = 0
        }
        
        return this.data[index]
    } 
    else if(type === 'side')
    {
        result = [];
        
        // check out of bounds
        if (index >= this.data[0][0].length) {
            index = this.data[0][0].length - 1
        }

        if (index < 0) {
            index = 0;
        }

        for(var x = 0 ; x < this.data.length; x++) {
            temp_array = []
            for(var y= 0; y < this.data[x].length; y++)
            {
                temp_array.push(this.data[x][y][index])
            }
            result.push(temp_array)
        }
        return result;
    } 
    else if(type === 'front') {
        
        result = [];

        for(var x = 0; x < this.data.length ; x++) {
            result.push(this.data[x][index])
        }
        return result;
    }
    return []
}

function draw(img_array, type='RGB', flipx=false, flipy=false, alpha=1) {
    
    var ctx = this.getContext('2d');
    
    var startx = (flipx)? img_array.length - 1 : 0;
    var endx = (flipx)? 0 : img_array.length - 1;
    var stepx = (flipx)? -1 : 1;

    var starty = (flipy)? img_array[0].length - 1 : 0;
    var endy = (flipy)? 0 : img_array[0].length -1;
    var stepy = (flipy)? -1 : 1;


    var imageWidth = img_array[0].length
    var imageHeight = img_array.length
    // change the size of the canvase depending on image array
    // TODO, use scale, its better to have a static sized canvas 
    // computation would be target / size of image array = scalar
    // scalar can be used to scale

    this.height = imageHeight;
    this.width = imageWidth;

    if(type === 'grayscale') {
        for (i=startx, x=0; i != endx; i+=stepx, x++){
            for(j=starty, y=0; j != endy;j+=stepy, y++){
                let color = img_array[i][j]
                ctx.fillStyle = "rgb("+color+","+color+","+color+")";
                ctx.fillRect(y, x, 1, 1);
            }
        }
    } 
    else if (type === 'RGB') {
                        
        for (i=startx, x=0; i != endx; i+=stepx, x++){
            for(j=starty, y=0; j != endy;j+=stepy, y++){
                let color = img_array[i][j]
                if(color !== 0)
                {
                    ctx.fillStyle = "rgba("+color['red']+","+color['green']+","+color['blue']+","+alpha+")";
                    ctx.fillRect(y, x, 1, 1);
                }
            }
        }
    }
}

// UTILITIES
function postRequest(url, data={}) {
    return fetch(url, {
        body: JSON.stringify(data),
        cache: 'no-cache',
        headers: {
            'content-type': 'application/json'
        },
        method: 'POST'
    })
}

main()