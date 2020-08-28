$(document).ready(function () {
    $("nav .nav-link").click(function() {
        console.log('d');
        $(this).addClass('active').siblings().removeClass('active');
    })


    $('#demonstration-file').on('change', function(e){
        $('.pd-card').css("display", "none");
        if (this.files && this.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('#result-image')
                    .attr('src', e.target.result);
            };

            reader.readAsDataURL(this.files[0]);
        }
    })

    $('#demonstration-form').on('submit', function(e) {
        e.preventDefault();
        if($('#demonstration-file')[0].files && $('#demonstration-file')[0].files[0])
         $('.pd-card').css("display", "inline-block");


        var file = $('#demonstration-file')[0].files[0];
        formdata = new FormData();
        formdata.append("img_data", file);
       // console.log($('#demonstration-file'));
        $.ajax({
            url : "demo/", // the endpoint
            type : "POST", // http method
            data : formdata, // data sent with the post request
            processData: false,
            contentType: false,
            // handle a successful response
            success : function(json) {
                $('#post-text').val(''); // remove the value from the input
              //  console.log(json); // log the returned json to the console
               // console.log("success"); // another sanity check
                var result = json["result"].split("___");
                var plant = result[0];
                var disease = result[1];
                disease = disease.replace("_", " ");
                $('#plant-category').html(plant);
                $('#plant-disease').html(disease);
                
                
                if(disease != "healthy")
                    $('#plant-disease').css("color", "#B22222");
            },
    
            // handle a non-successful response
            error : function(xhr,errmsg,err) {
                console.log('error');
              }
        });
    })



});
