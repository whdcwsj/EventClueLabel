var scaleMap = {
    "1":"小规模",
    "2":"中规模",
    "3":"大规模"
};

var influenceMap = {
    "1":"正向",
    "2":"负向"
};

var strengthMap = {
    "1":"低强度",
    "2":"中强度",
    "3":"高强度"
};

var degreeMap = {
    "1":"小",
    "2":"中",
    "3":"高",
    "4":"最高"
};

$("#spider").click(function(){
    var textid = $("#category").val();
    var mytext = $("#inputtext").val();
    console.log(textid, mytext);
    var data = {
        "primaryClassification": textid,
        "eventName": mytext,
        "id": "wsj"
    };
    // 将一个JavaScript 对象或值转换为JSON 字符串
    data = JSON.stringify(data);
    console.log(data);
    $.ajax({
        url:"http://118.195.234.43:12456/eventCluelabel/predict",
        type:"post",
        data: data, 
        success: function(data, status){
            console.log("success")
            console.log(data)
            var result = JSON.stringify(data);
            console.log(result);
            $("#score").val(data.score);
            console.log(data.score);
            console.log(data["scale "]);
            console.log(data.strength);
            if(textid==2){
                $("#scale").val(scaleMap[data["scale "]]);
                $("#strength").val(strengthMap[data.strength]);
            }else{
                $("#scale").val(influenceMap[data["scale "]]);
                $("#strength").val(degreeMap[data.strength]);
            }
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log("fail")
            // console.log(jqXHR)
            // console.log(textStatus)
            // console.log(errorThrown)
            // console.log(jqXHR.responseText)
        },
        // contentType: 'application/json',
        contentType:"application/json;charset=utf-8",
        dataType:"json"
    })
});

// 对应的五个答案为: score 5, scale 3, influence 不存在, strength 3, degree不存在
// primaryClassification=2
// 俄罗斯总统新闻秘书佩斯科夫表示，北约已经事实上卷入俄乌冲突，但俄罗斯将把对乌克兰的“特别军事行动”进行到底。

// 对应的五个答案为: score -3, scale 不存在, influence 1, strength 不存在, degree 1
// primaryClassification=4
// 德国最大租车公司西克斯特（Sixt）计划在今后6年内采购10万辆中国厂商比亚迪生产的电动汽车。

// 对应的五个答案为: score -5, scale 2, influence 不存在, strength 2, degree 不存在
// primaryClassification=2
// 我空军轰-6轰炸机、空警-2000预警机、运-8电子干扰机、图-154电子侦察机以及苏-35、歼-11战机护航编队等组成两个打击集群，分别从台湾的南方和北方两个不同的方向，同时并进完成绕岛巡航。