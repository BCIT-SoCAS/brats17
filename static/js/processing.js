console.log("ready");

let bar = document.getElementById("bar");
let loader = document.getElementById("loader");
let status = document.getElementById("status");

let query_interval;

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


function queryStatus() {
    postRequest('/queryStatus')
        .then(response=>response.json())
        .then(data => {
        
        bar.innerHTML = data
        statusUpd(data);
        
        if (data == 'Done') {
            clearInterval(query_interval)
            
            location.href="/brain_view"
        }
    })
        .catch(err => console.log(err));
}

function statusUpd() {
    
    let width = "0";
    let marginRight = "0%";
    counter = 1
    switch(data) {
        case 'Configuring Neural Network':
            width = '15%';
            marginRight = "-15%"
            counter = 1;
            break;
        case 'Loading Models':
            width = '50%';
            marginRight = '-50%';
            counter = 2;
            break;
        case 'Loading Data':
            width = '65%';
            marginRight = '-65%';
            counter = 3;
            break;
        case  'Testing Data on Models':
            width = '90%'
            marginRight = '-90%';
            counter = 4;
            break;
        case 'Done':
            width = '100%';
            marginRight = '-100%';
            counter = 5
        
    }
    
    document.getElementById("progress").style.width = width;
    document.getElementById("progress").style.marginRight = marginRight;
    document.getElementById("image").setAttribute("class", "gif0" + counter);
}
query_interval = setInterval(queryStatus, 4000);