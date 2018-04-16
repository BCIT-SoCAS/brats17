console.log("ready");

let bar = document.getElementById("bar");
let loader = document.getElementById("loader");
let status = document.getElementById("status");
let counter = 1;
let loadProg = 0;

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
        statusUpd();
        
        if (data == 'Done') {
            clearInterval(query_interval)
            
            // location.href="/brain_view"
        }
    })
        .catch(err => console.log(err));
}

function statusUpd() {
    counter++;
    if(counter > 4) {
        counter = 1;
    }
    loadProg++;
    
    if (loadProg == 1) {
        document.getElementById("progress").style.width = "25%";
        document.getElementById("progress").style.marginRight = "-25%";
    }
    else if (loadProg == 2) {
        document.getElementById("progress").style.width = "50%";
        document.getElementById("progress").style.marginRight = "-50%";
    }
    else if (loadProg == 3) {
        document.getElementById("progress").style.width = "75%";
        document.getElementById("progress").style.marginRight = "-75%";
    }
    else if (loadProg == 4) {
        document.getElementById("progress").style.width = "100%";
        document.getElementById("progress").style.marginRight = "-100%";
    }
    
    document.getElementById("image").setAttribute("class", "gif0" + counter);
}
query_interval = setInterval(queryStatus, 4000);