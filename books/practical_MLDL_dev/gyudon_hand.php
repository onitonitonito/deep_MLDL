<?php
  $self = "gyudon-hand.php";
  $base_dir = dirname(__FILE__)."/image";
  $unknown_dir = "gyudon";
  $dir = array(
    "일반규동" => "normal",
    "생강규동" => "beni",
    "양파규동" => "negi",
    "치즈규동" => "cheese",
    "김치규동" => "kimuti",
    "기타" => "other",
  );

// 필요한 폴더 생성하기
foreach ($dirs as $key => $dir) {
  $path = $base_dir."/$dir";
  if (!file_exists($path))  {
    mkdir($path);
    chmod($path, 0777);
  }
}

// 분류하거나 입력양식 제공하기
$m = isset($_GET["m"]) ? $_GET["m"] : "";

if ($m == "mv")  {                // 분류하기
  $target = $_GET["target"];      // 요청 매개변수 추출하기
  $to = $_GET["to"];
  $path = $base_dir."/$unknown_dir/$target";

  //요청 매개변수 확인하기
  if ($target == "")  {
    echo "error...";
    exit;
  }

  if (!file_exists($path))  {
    echo "<a href='$self'>already... </a>";
    exit;
  }

  if (!file_exists($base_dir/$to))  {
    echo "system error : no dir $to";
    exit;
  }

  $path_to = "$base_dir/$to/$target";
  copy($path, $path_to);

  if (file_exists($path_to))  {
    unlink($path);
  } else {
    echo "Sorry, could not move.";
    exit;
  }


  /* ---------- 이 부분이 모호하다 -------------- */
  // 선택화면의 리다이렉트

    header("location: $self");
    echo "<a href='$self'>Thank you, moved.</a>";
} else {                                        // 규동 선택 입력양식 만들기
  $files = glob("$base_dir/$unknown_dir/*.jpg");  // 이미지 가져오기

  if (count($files) == 0)  {
    echo "<h1>완료</h1>";
    exit;
  }

  shuffle($files);      // 적당한 화일을 선택합니다.
  $target = basename($files[0]);
  $remain = count($files);        // 남은 화일 수
  $buttons = "";      // 선택지 생성하기

  foreach ($dirs as $key => $dir) {
    # code...
    $fs = glob("$base_dir/$dir/*.jpg");     // 분류한 화일 수
    $cnt = count($fs);
    $api = "$self?m=mv&target=$target&to=$dir";
    $buttons .= "[<a href='$api'>$key($cnt)</a>]";
  }

  echo <<< EOS
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=320px">
        <style>
          body  {
            text-align : center;
            font-sizze : 24px;
          }
        </style>
      </head>

      <body>

      </body>
    </html>
  EOS;

  }
?>
