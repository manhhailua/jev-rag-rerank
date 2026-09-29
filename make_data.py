"""Sinh data/sample_inputs.json — queries + chunks (512-token, đo bằng tiktoken) + ground truth.

Chạy: uv run --with tiktoken python3 make_data.py
"""
import json
import os

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def nt(s):
    if ENC is None:
        return len(s) // 4
    return len(ENC.encode(s))

# Mỗi topic: 1 query ngắn (<=20 tokens) + 2 chunks (~512 tokens) + extra (câu bổ sung).
TOPICS = [
{
 "id":"sql-index", "name":"Tối ưu SQL bằng index",
 "query":"Dùng index tăng tốc SQL PostgreSQL thế nào?",
 "chunks":[
  "Chỉ mục (index) trong PostgreSQL là cấu trúc dữ liệu giúp hệ quản trị cơ sở dữ liệu định vị các dòng dữ liệu mà không cần quét toàn bộ bảng. Khi bạn tạo một chỉ mục B-tree trên một cột, PostgreSQL xây dựng một cây cân bằng lưu giữ các giá trị đã sắp xếp cùng con trỏ trỏ tới vị trí vật lý của từng dòng. Nhờ vậy, các truy vấn dùng mệnh đề WHERE, JOIN hoặc ORDER BY trên cột đó có thể dùng phép tìm kiếm nhị phân thay vì Seq Scan, giúp giảm độ phức tạp từ O(n) xuống O(log n). Tuy nhiên, chỉ mục không miễn phí: mỗi lần ghi dữ liệu hệ quản trị phải cập nhật chỉ mục, làm chậm thao tác ghi và tăng dung lượng lưu trữ. Vì vậy, nguyên tắc chung là chỉ tạo chỉ mục trên các cột thực sự được dùng để lọc hoặc nối dữ liệu thường xuyên. Với các truy vấn kết hợp nhiều điều kiện, một chỉ mục tổng hợp đặt cột có độ chọn lọc cao lên đầu sẽ phát huy hiệu quả tốt hơn.",
  "Ngoài B-tree mặc định, PostgreSQL còn cung cấp nhiều loại chỉ mục chuyên dụng phù hợp từng kiểu truy vấn. Chỉ mục GIN và GiST hỗ trợ tìm kiếm toàn văn bản và kiểu dữ liệu hình học, trong khi chỉ mục BRIN hiệu quả với các bảng rất lớn có cột được sắp theo thứ tự vật lý. Chỉ mục Hash phù hợp cho các truy vấn so sánh bằng, còn chỉ mục phủ dùng INCLUDE để lưu thêm các cột không thuộc khóa, cho phép truy vấn trả kết quả ngay từ chỉ mục mà không cần truy cập heap. Khi một truy vấn chỉ chạm vào các cột đã được phủ bởi chỉ mục, PostgreSQL có thể dùng Index Only Scan, bỏ qua bước đọc bảng và tăng tốc đáng kể. Việc lựa chọn đúng loại chỉ mục phụ thuộc vào đặc điểm dữ liệu và mẫu truy vấn thực tế, nên cần phân tích kỹ trước khi áp dụng. Bên cạnh đó, chỉ mục cần được bảo trì định kỳ bằng lệnh REINDEX khi bị phân mảnh nặng sau nhiều lần cập nhật.",
 ],
 "extra":[
  "Chỉ mục một phần (partial index) chỉ đánh chỉ mục cho các dòng thỏa một điều kiện lọc, giúp giảm kích thước chỉ mục khi phần lớn dữ liệu không cần tra cứu thường xuyên.",
  "Nên tránh dùng hàm trên cột trong mệnh đề WHERE, vì điều đó khiến planner không thể tận dụng chỉ mục thông thường trên cột gốc.",
  "Chỉ mục biểu thức (expression index) đánh chỉ mục kết quả của một biểu thức thay vì giá trị cột gốc, hữu ích khi truy vấn luôn lọc theo biểu thức đó.",
  "Bảng pg_stat_user_indexes cho biết tần suất mỗi chỉ mục được dùng, giúp phát hiện chỉ mục thừa để dọn dẹp.",
  "Khi tạo chỉ mục trên bảng lớn đang phục vụ, nên dùng CREATE INDEX CONCURRENTLY để tránh khóa ghi trong quá trình xây dựng.",
  "Mỗi chỉ mục tăng thêm đều làm chậm thao tác ghi, nên cần cân nhắc giữa lợi ích đọc và chi phí ghi trước khi thêm chỉ mục mới.",
 ],
},
{
 "id":"sql-vacuum", "name":"VACUUM và MVCC",
 "query":"VACUUM và ANALYZE dùng làm gì?",
 "chunks":[
  "PostgreSQL dùng cơ chế điều khiển tương tranh đa phiên bản (MVCC) để cho phép nhiều giao dịch đọc và ghi đồng thời mà không chặn nhau. Theo cơ chế này, khi một dòng được cập nhật hoặc xóa, phiên bản cũ không bị ghi đè ngay mà được đánh dấu là dead tuple và bản ghi mới được chèn vào. Hệ quả là bảng và chỉ mục dần tích tụ các bản ghi chết, gây phình to dữ liệu và làm chậm các truy vấn quét bảng. Lệnh VACUUM có nhiệm vụ dọn dẹp các dead tuple này, đánh dấu lại không gian để tái sử dụng và cập nhật bản đồ hiển thị giúp các lần quét sau bỏ qua các trang đã sạch. Nếu không chạy VACUUM định kỳ, bảng sẽ ngày càng phình to, thống kê trở nên lỗi thời và hiệu năng suy giảm rõ rệt theo thời gian.",
  "Lệnh ANALYZE có vai trò khác với VACUUM: nó thu thập thống kê về phân bố giá trị trong các cột của bảng và lưu vào bảng pg_statistic. Query planner dựa vào những thống kê này để ước lượng số dòng trả về của từng bước trong kế hoạch thực thi và chọn chiến lược nối, thứ tự quét, hay quyết định có nên dùng chỉ mục hay không. Nếu thống kê đã cũ hoặc không chính xác, planner có thể chọn một kế hoạch kém tối ưu, chẳng hạn dùng Seq Scan cho một bảng có hàng triệu dòng trong khi chỉ cần trả về vài chục dòng. Do đó, sau khi nạp một lượng lớn dữ liệu hoặc thay đổi cấu trúc bảng, bạn nên chạy ANALYZE để cập nhật thống kê trước khi đánh giá hiệu năng.",
 ],
 "extra":[
  "Autovacuum có thể được điều chỉnh qua các tham số như autovacuum_vacuum_scale_factor để kích hoạt sớm hơn trên bảng có nhiều biến động.",
  "Bảng thường xuyên cập nhật hoặc xóa cần VACUUM thường xuyên hơn bảng chỉ đọc, vì chúng sinh ra nhiều dead tuple hơn.",
  "Lệnh VACUUM FULL viết lại toàn bộ bảng và trả không gian về hệ điều hành nhưng lại khóa bảng trong thời gian thực thi.",
  "pg_stat_user_tables lưu số dead tuple và thời điểm VACUUM gần nhất, giúp theo dõi tình trạng phình bảng theo thời gian.",
  "Thống kê lỗi thời khiến planner ước lượng sai số dòng, dẫn tới chọn kế hoạch thực thi kém hiệu quả một cách khó phát hiện.",
  "Sau khi xóa hoặc cập nhật hàng loạt, chạy VACUUM giúp trả lại không gian trống để bảng có thể tái sử dụng cho các bản ghi mới.",
 ],
},
{
 "id":"raft-consensus", "name":"Thuật toán đồng thuận Raft",
 "query":"Thuật toán Raft hoạt động thế nào?",
 "chunks":[
  "Raft là một thuật toán đồng thuận được thiết kế để dễ hiểu hơn Paxos, nhằm giúp một nhóm máy chủ duy trì một nhật ký lệnh nhất quán ngay cả khi có sự cố. Các nút trong cụm Raft đảm nhiệm một trong ba vai trò: leader, follower hoặc candidate. Leader chịu trách nhiệm nhận yêu cầu từ client, ghi vào nhật ký và nhân bản sang các follower. Để chọn leader, Raft dùng cơ chế bầu cử: khi một follower không nhận được nhịp tim từ leader trong một khoảng thời gian chờ ngẫu nhiên, nó chuyển thành candidate và yêu cầu bỏ phiếu từ các nút khác. Nút nhận được đa số phiếu sẽ trở thành leader mới. Việc dùng khoảng chờ ngẫu nhiên giúp tránh tình trạng hai nút đồng thời tranh cử dẫn đến bế tắc.",
  "Để đảm bảo an toàn, Raft yêu cầu mọi mục nhật ký phải được ghi bền vững trên đa số nút trước khi áp dụng vào máy trạng thái. Leader chỉ cam kết một mục khi nó đã được nhân bản tới quá nửa số nút trong cụm, nhờ đó hệ thống vẫn nhất quán nếu leader gặp sự cố ngay sau khi cam kết. Thuộc tính bầu cử đảm bảo nút được chọn làm leader luôn có nhật ký cập nhật nhất, vì lá phiếu chỉ trao cho candidate có nhật ký không cũ hơn người bỏ phiếu. Khi một nút khởi động lại sau sự cố, nó sẽ nhận các mục còn thiếu từ leader thông qua cơ chế cập nhật nhật ký.",
 ],
 "extra":[
  "Nhịp tim được leader gửi định kỳ để duy trì quyền lãnh đạo và ngăn các follower khởi động bầu cử không cần thiết.",
  "Term (nhiệm kỳ) là số hiệu tăng dần theo mỗi vòng bầu cử, dùng để phân biệt các leader khác nhau trong thời gian chạy.",
  "Mỗi mục nhật ký gồm lệnh, số thứ tự và số nhiệm kỳ, giúp phát hiện các mục lỗi thời bị ghi bởi leader cũ.",
  "Raft chỉ cần đa số nút sống sót để tiếp tục hoạt động, nên một cụm ba nút chịu được một nút hỏng.",
  "Sau khi bầu leader mới, các follower bị tụt hậu sẽ được leader đồng bộ lại các mục nhật ký còn thiếu.",
  "etcd và Consul là những hệ thống nổi tiếng dùng Raft làm nền tảng cho lưu trữ nhất quán phân tán.",
 ],
},
{
 "id":"docker", "name":"Container và Docker",
 "query":"Docker image khác container thế nào?",
 "chunks":[
  "Docker image là một khuôn mẫu bất biến chứa mọi thứ cần thiết để chạy một ứng dụng, bao gồm mã nguồn, thư viện phụ thuộc, biến môi trường và các tệp cấu hình. Image được xây dựng từng lớp thông qua Dockerfile, trong đó mỗi lệnh như FROM, RUN hay COPY tạo ra một lớp mới chồng lên lớp trước. Các lớp này có thể được chia sẻ và tái sử dụng giữa nhiều image khác nhau, giúp tiết kiệm dung lượng đĩa và băng thông khi tải về. Khi bạn chạy một image bằng lệnh docker run, Docker tạo ra một container là phiên bản đang hoạt động của image, có thêm một lớp ghi mỏng trên cùng để lưu các thay đổi trong quá trình chạy.",
  "Vì image là bất biến, mọi thay đổi bên trong container đều biến mất khi container bị xóa, trừ khi bạn gắn volume hoặc bind mount để lưu dữ liệu ra ngoài. Đây là nguyên tắc quan trọng khi thiết kế ứng dụng: trạng thái cần được lưu ở nơi bền vững, còn container có thể bị hủy và tạo lại bất cứ lúc nào. Việc tối ưu kích thước image cũng là một chủ đề được quan tâm, chẳng hạn dùng image nền nhỏ gọn như alpine, gộp nhiều lệnh RUN vào một để giảm số lớp, và dùng multi-stage build để loại bỏ công cụ biên dịch khỏi image cuối cùng. Một image nhỏ giúp tải nhanh hơn, khởi động nhanh hơn và giảm bề mặt tấn công.",
 ],
 "extra":[
  "Container chia sẻ nhân hệ điều hành với máy chủ nhưng bị cô lập về tài nguyên qua cơ chế namespace và cgroup.",
  "Nhiều container có thể chạy đồng thời trên cùng một máy mà không xung đột nhờ sự cô lập này.",
  "Docker Hub là registry công khai phổ biến để chia sẻ image, còn doanh nghiệp thường dùng registry riêng để kiểm soát bảo mật.",
  "Lớp ghi mỏng của container bị xóa khi container dừng, nên dữ liệu quan trọng phải được lưu qua volume.",
  "Multi-stage build giúp tách biệt môi trường biên dịch và môi trường chạy, cho image cuối gọn và an toàn hơn.",
  "Image được xây từ Dockerfile có thể tái tạo chính xác trên mọi máy, giúp tránh lỗi do khác biệt môi trường.",
 ],
},
{
 "id":"kubernetes", "name":"Kubernetes và điều phối container",
 "query":"Pod khác Deployment thế nào?",
 "chunks":[
  "Kubernetes là nền tảng mã nguồn mở dùng để tự động triển khai, mở rộng và quản lý các ứng dụng container. Đơn vị nhỏ nhất mà Kubernetes quản lý là Pod, một nhóm gồm một hoặc nhiều container được lập lịch chạy chung trên cùng một nút, chia sẻ mạng và không gian lưu trữ. Các container trong cùng Pod có thể giao tiếp với nhau qua localhost và thường được thiết kế để đảm nhiệm các vai trò phụ trợ như thu thập log hay xử lý sidecar. Pod về bản chất là tạm thời, có thể bị thay thế bất cứ lúc nào khi nút gặp sự cố hoặc khi hệ thống mở rộng. Vì vậy, ứng dụng không nên phụ thuộc vào địa chỉ của một Pod cụ thể mà cần một lớp trừu tượng cao hơn để quản lý vòng đời và số lượng bản sao.",
  "Deployment là một controller cấp cao hơn, quản lý một tập hợp các Pod giống hệt nhau và đảm bảo số lượng bản sao mong muốn luôn được duy trì. Khi bạn khai báo một Deployment với ba bản sao, controller sẽ liên tục theo dõi và tự động tạo Pod mới nếu có Pod bị xóa hoặc hỏng. Deployment cũng hỗ trợ cập nhật cuốn chiếu (rolling update) và quay lui (rollback), cho phép triển khai phiên bản mới mà không gây gián đoạn dịch vụ. Đứng sau Pod còn có Service, một đối tượng cung cấp địa chỉ ổn định và cân bằng tải tới các Pod, giúp client không cần biết Pod thực tế nằm ở đâu.",
 ],
 "extra":[
  "Nhờ tách bạch trách nhiệm giữa Pod, Deployment và Service, Kubernetes vận hành hệ thống phân tán một cách có kiểm soát.",
  "ReplicaSet là đối tượng trung gian mà Deployment quản lý, đảm bảo đúng số lượng Pod chạy tại mọi thời điểm.",
  "Sidecar là container phụ chạy cùng Pod, thường đảm nhiệm thu thập log, proxy mạng hoặc đồng bộ cấu hình.",
  "Khi cập nhật rolling update, Deployment thay thế từng Pod một để dịch vụ không bị ngừng hoàn toàn.",
  "Service dùng label selector để chọn tập Pod cần cân bằng tải, thay vì trỏ tới địa chỉ IP cố định của Pod.",
  "Khả năng tự phục hồi giúp Kubernetes tạo lại Pod khi nó bị hỏng, nhưng không khôi phục dữ liệu không được lưu bền.",
 ],
},
{
 "id":"tcp-http", "name":"TCP và HTTP keep-alive",
 "query":"HTTP keep-alive giảm độ trễ thế nào?",
 "chunks":[
  "TCP là giao thức truyền tải hướng kết nối, đảm bảo dữ liệu đến đích đúng thứ tự và không bị mất mát thông qua cơ chế bắt tay ba bước và xác nhận từng gói tin. Trước khi truyền dữ liệu, hai bên phải thực hiện quá trình bắt tay gồm ba gói SYN, SYN-ACK và ACK để thiết lập kết nối, mất ít nhất một lần khứ hồi. Khi kết thúc, kết nối được đóng bằng bắt tay bốn bước. Trong mô hình HTTP phiên bản đầu, mỗi yêu cầu tài nguyên lại mở một kết nối TCP mới, nghĩa là trang web có nhiều ảnh và tệp sẽ phải chịu chi phí bắt tay lặp đi lặp lại, làm tăng độ trễ đáng kể.",
  "HTTP keep-alive (còn gọi là kết nối bền vững) cho phép tái sử dụng cùng một kết nối TCP cho nhiều yêu cầu và phản hồi, tránh phải bắt tay lại cho từng tài nguyên. Nhờ đó, trình duyệt có thể tải nhiều đối tượng của một trang web chỉ qua một kết nối, giảm rõ rệt độ trễ và tải trên máy chủ. Cơ chế này được mặc định bật trong HTTP/1.1, với thời gian giữ kết nối do tiêu đề Keep-Alive và Timeout quy định. HTTP/2 tiến xa hơn bằng cách ghép kênh nhiều yêu cầu trên cùng một kết nối, cho phép các yêu cầu được xử lý song song mà không bị chặn lẫn nhau.",
 ],
 "extra":[
  "Chi phí bắt tay TCP càng đáng kể trên các kết nối có độ trễ mạng cao hoặc máy chủ ở xa về mặt địa lý.",
  "Giữ kết nối lâu cũng tiêu tốn tài nguyên máy chủ, nên cần cân bằng giữa giảm độ trễ và chi phí duy trì kết nối nhàn rỗi.",
  "Tiêu đề Keep-Alive cho biết thời gian tối đa một kết nối được giữ mở trước khi máy chủ đóng.",
  "Trong HTTP/1.0, keep-alive không bật mặc định và phải được yêu cầu tường minh qua tiêu đề Connection: keep-alive.",
  "HTTP/2 loại bỏ vấn đề chặn đầu dòng (head-of-line blocking) ở mức HTTP nhờ ghép kênh nhiều luồng.",
  "Việc tái sử dụng kết nối còn giảm tải CPU do không phải thiết lập và hủy socket liên tục.",
 ],
},
{
 "id":"tls", "name":"TLS và chứng chỉ số",
 "query":"Bắt tay TLS diễn ra thế nào?",
 "chunks":[
  "TLS (Transport Layer Security) là giao thức mã hóa bảo vệ dữ liệu truyền qua mạng, được dùng rộng rãi trong HTTPS để đảm bảo tính bí mật và toàn vẹn của thông tin. Quá trình bắt tay TLS bắt đầu khi client gửi lời chào chứa danh sách các bộ mã hóa và phiên bản giao thức mà nó hỗ trợ. Máy chủ phản hồi bằng việc chọn một bộ mã hóa, gửi chứng chỉ số của mình và yêu cầu client xác thực. Client kiểm tra chứng chỉ dựa trên chuỗi tin cậy từ các tổ chức phát hành chứng chỉ (CA) đã được cài sẵn trong hệ điều hành. Sau khi xác thực thành công, hai bên trao đổi khóa phiên thông qua cơ chế trao đổi khóa như ECDHE, tạo ra một khóa đối xứng dùng chung để mã hóa dữ liệu cho phần còn lại của phiên làm việc.",
  "Trong TLS 1.3, quá trình bắt tay được rút gọn chỉ còn một lần khứ hồi thay vì hai lần như các phiên bản trước, nhờ loại bỏ các thuật toán cũ không an toàn và cho phép client gửi ngay khóa chia sẻ dựa trên nhóm đường cong elliptic. Điều này giảm độ trễ thiết lập kết nối đáng kể, đặc biệt quan trọng với các ứng dụng nhạy cảm về tốc độ phản hồi. Chứng chỉ số đóng vai trò trung tâm trong việc xác thực danh tính máy chủ, ngăn chặn tấn công trung gian (man-in-the-middle) bằng cách ràng buộc khóa công khai với tên miền.",
 ],
 "extra":[
  "Việc gia hạn chứng chỉ đúng hạn và cấu hình đúng chuỗi tin cậy là yêu cầu vận hành cơ bản của một hệ thống HTTPS.",
  "Một chứng chỉ hết hạn sẽ khiến trình duyệt chặn truy cập và hiển thị cảnh báo kết nối không an toàn.",
  "ECDHE cho phép trao đổi khóa với tính chất perfect forward secrecy, bảo vệ phiên cũ ngay cả khi khóa máy chủ bị lộ.",
  "Chuỗi tin cậy gồm chứng chỉ máy chủ, chứng chỉ trung gian và chứng chỉ gốc của tổ chức phát hành.",
  "Tấn công trung gian bị ngăn chặn nhờ client xác minh chữ ký và tên miền trong chứng chỉ trước khi tin tưởng.",
  "Sau bắt tay, dữ liệu được mã hóa bằng khóa đối xứng vì nhanh hơn nhiều so với mã hóa bất đối xứng.",
 ],
},
{
 "id":"ml-overfit", "name":"Overfitting trong học máy",
 "query":"Cách ngăn chặn overfitting?",
 "chunks":[
  "Overfitting xảy ra khi một mô hình học quá khớp với dữ liệu huấn luyện, bao gồm cả nhiễu và các ngoại lệ, khiến nó hoạt động tốt trên tập huấn luyện nhưng kém trên dữ liệu chưa từng thấy. Dấu hiệu điển hình là khoảng cách lớn giữa độ chính xác trên tập huấn luyện và tập kiểm tra. Nguyên nhân thường gặp gồm mô hình quá phức tạp so với lượng dữ liệu, huấn luyện quá nhiều epoch, hoặc dữ liệu có quá nhiều đặc trưng không liên quan. Để chẩn đoán, người ta thường theo dõi đường cong học tập và dùng một tập xác thực riêng để phát hiện thời điểm mô hình bắt đầu học vẹt. Việc chia dữ liệu thành các tập huấn luyện, xác thực và kiểm tra một cách ngẫu nhiên nhưng có tầng lớp giúp đánh giá khách quan khả năng tổng quát hóa của mô hình.",
  "Có nhiều kỹ thuật giảm overfitting, trong đó phổ biến nhất là chính quy hóa (regularization) nhằm phạt các trọng số lớn, như chuẩn L1 và L2. Kỹ thuật dropout tắt ngẫu nhiên một phần nơ-ron trong quá trình huấn luyện, buộc mạng phải học các biểu diễn bền vững hơn thay vì phụ thuộc vào một vài nơ-ron cụ thể. Dừng sớm (early stopping) ngừng huấn luyện khi độ lỗi trên tập xác thực bắt đầu tăng, ngăn mô hình học quá mức. Tăng cường dữ liệu (data augmentation) tạo thêm các biến thể từ dữ liệu gốc để làm giàu tập huấn luyện, đặc biệt hiệu quả trong thị giác máy tính.",
 ],
 "extra":[
  "Đơn giản hóa mô hình bằng cách giảm số tham số hoặc số lớp cũng hiệu quả khi lượng dữ liệu hạn chế.",
  "Kết hợp nhiều kỹ thuật thường cho khả năng tổng quát hóa tốt hơn so với chỉ dùng một biện pháp đơn lẻ.",
  "Chuẩn L1 có xu hướng đưa một số trọng số về không, tạo mô hình thưa thớt và tự chọn đặc trưng.",
  "Chuẩn L2 phạt bình phương trọng số, giữ các trọng số nhỏ nhưng hiếm khi bằng không hoàn toàn.",
  "Dropout thường được tắt trong lúc suy luận để dùng toàn bộ mạng thay vì lấy mẫu ngẫu nhiên.",
  "Đường cong học tập tách xa giữa train và validation là tín hiệu rõ ràng của overfitting.",
 ],
},
{
 "id":"jwt", "name":"JWT và xác thực token",
 "query":"JWT là gì, lưu token ở đâu?",
 "chunks":[
  "JSON Web Token (JWT) là một chuẩn mở dùng để truyền tải thông tin giữa các bên dưới dạng một đối tượng JSON được ký số, thường được dùng trong xác thực và ủy quyền. Một JWT gồm ba phần được phân tách bằng dấu chấm: header mô tả thuật toán ký, payload chứa các claim như danh tính người dùng và thời gian hết hạn, và signature dùng để xác minh tính toàn vẹn. Điểm mạnh của JWT là tính không trạng thái, cho phép máy chủ xác thực yêu cầu mà không cần lưu phiên ở phía server, phù hợp với kiến trúc phân tán và microservices. Tuy nhiên, payload của JWT chỉ được mã hóa dạng Base64 chứ không được mã hóa thực sự, nên bất kỳ ai có token đều đọc được nội dung.",
  "Vấn đề lưu trữ JWT ở phía client cũng quan trọng không kém. Lưu token trong localStorage tuy tiện lợi nhưng dễ bị đánh cắp qua tấn công XSS, vì bất kỳ script nào chạy trong trang đều có thể đọc localStorage. Phương án an toàn hơn là lưu token trong cookie có cờ HttpOnly, khiến JavaScript không thể truy cập, kết hợp cờ Secure để chỉ gửi qua HTTPS và cờ SameSite để giảm nguy cơ tấn công CSRF. Bên cạnh đó, cần đặt thời gian hết hạn hợp lý và cung cấp cơ chế làm mới token thông qua refresh token được lưu riêng biệt.",
 ],
 "extra":[
  "Tuyệt đối không đặt thông tin nhạy cảm như mật khẩu vào payload JWT vì payload có thể bị đọc dễ dàng.",
  "Luôn dùng HTTPS để bảo vệ token khỏi bị chặn bắt trong quá trình truyền qua mạng.",
  "Khi token bị lộ, khả năng thu hồi là hạn chế cố hữu của JWT vì bản chất không trạng thái của nó.",
  "Một số hệ thống dùng danh sách chặn hoặc chuyển sang token dạng mờ để có thể thu hồi tức thời từ phía server.",
  "Cờ SameSite=Strict hoặc Lax giúp giảm nguy cơ tấn công CSRF lên cookie lưu token.",
  "Refresh token thường có thời gian sống dài hơn access token và được lưu ở nơi an toàn hơn.",
 ],
},
{
 "id":"owasp", "name":"OWASP Top 10 và bảo mật ứng dụng",
 "query":"Ngăn SQL injection và XSS thế nào?",
 "chunks":[
  "SQL injection là một trong những lỗ hổng bảo mật ứng dụng web nguy hiểm nhất, xảy ra khi kẻ tấn công chèn mã SQL độc hại vào dữ liệu đầu vào được ghép trực tiếp vào câu truy vấn. Ví dụ, nếu ứng dụng nối chuỗi để xây dựng câu lệnh như lấy tên người dùng từ biểu mẫu rồi chèn trực tiếp, kẻ tấn công có thể thay đổi cấu trúc câu lệnh để đọc toàn bộ bảng hoặc xóa dữ liệu. Cách phòng chống hiệu quả nhất là dùng tham số hóa truy vấn (parameterized query) hoặc các câu lệnh prepared statement, giúp tách biệt dữ liệu khỏi mã SQL nên dữ liệu không bao giờ bị diễn giải thành lệnh. Ngoài ra, nguyên tắc đặc quyền tối thiểu khuyến nghị tài khoản cơ sở dữ liệu của ứng dụng chỉ nên có các quyền cần thiết, nhằm giới hạn thiệt hại nếu lỗ hổng bị khai thác.",
  "Cross-site scripting (XSS) xảy ra khi ứng dụng hiển thị dữ liệu do người dùng cung cấp mà không lọc hoặc mã hóa, cho phép kẻ tấn công chèn script chạy trên trình duyệt của nạn nhân. XSS có thể bị lợi dụng để đánh cắp cookie phiên, giả mạo nội dung hoặc thực hiện hành động thay mặt người dùng. Biện pháp phòng chống cốt lõi là mã hóa đầu ra theo đúng ngữ cảnh: mã hóa HTML khi đưa dữ liệu vào nội dung trang, mã hóa thuộc tính khi đặt vào thẻ, và mã hóa JavaScript khi đặt vào script. Sử dụng các framework hiện đại vốn có cơ chế thoát tự động cũng giúp giảm rủi ro đáng kể.",
 ],
 "extra":[
  "Chính sách bảo mật nội dung (CSP) là lớp phòng thủ thứ hai bằng cách hạn chế nguồn script mà trình duyệt được phép thực thi.",
  "Rà soát mã và kiểm thử xâm nhập định kỳ giúp phát hiện sớm các điểm nối chuỗi truy vấn còn sót.",
  "Mã hóa đầu ra phải đúng ngữ cảnh: HTML, thuộc tính và JavaScript mỗi ngữ cảnh có quy tắc mã hóa riêng.",
  "OWASP Top 10 là danh sách các rủi ro bảo mật ứng dụng phổ biến được cập nhật định kỳ.",
  "Broken Access Control là hạng mục thường xuyên đứng đầu OWASP Top 10 trong các phiên bản gần đây.",
  "Kết hợp nhiều lớp phòng thủ giúp giảm thiểu đáng kể nguy cơ từ cả SQL injection lẫn XSS.",
 ],
},
{
 "id":"git-ci", "name":"Git flow và CI/CD",
 "query":"CI/CD pipeline có những bước nào?",
 "chunks":[
  "Tích hợp liên tục (CI) là thực hành tự động xây dựng và kiểm thử mã mỗi khi có thay đổi được đẩy lên kho lưu trữ, nhằm phát hiện lỗi sớm trong vòng đời phát triển. Khi một nhà phát triển tạo yêu cầu gộp nhánh, hệ thống CI sẽ tự động lấy mã mới, cài đặt phụ thuộc, chạy bộ kiểm thử tự động và báo cáo kết quả. Việc này giúp giảm xung đột tích hợp vốn thường xảy ra khi nhiều người cùng làm việc trên một codebase trong thời gian dài. Các công cụ phổ biến như GitHub Actions, GitLab CI hay Jenkins cung cấp cách định nghĩa pipeline dưới dạng mã, cho phép tái tạo quy trình một cách nhất quán. Một bộ kiểm thử tự động đáng tin cậy là nền tảng của CI, vì nếu kiểm thử không phản ánh đúng hành vi mong muốn thì việc tự động hóa sẽ mất đi ý nghĩa.",
  "Triển khai liên tục (CD) mở rộng CI bằng cách tự động đưa mã đã kiểm thử lên môi trường chạy thật sau khi vượt qua các bước xác thực. Pipeline CD thường gồm các giai đoạn như đóng gói ứng dụng thành image, triển khai lên môi trường staging để kiểm thử thêm, rồi phát hành ra production với cơ chế quay lui nếu có sự cố. Chiến lược triển khai có thể là blue-green, canary hoặc rolling update để giảm rủi ro gián đoạn dịch vụ. Việc tự động hóa toàn bộ chuỗi từ commit tới production giúp rút ngắn thời gian đưa tính năng ra thị trường và giảm sai sót thủ công.",
 ],
 "extra":[
  "CD đòi hỏi hạ tầng giám sát tốt để phát hiện nhanh các hồi quy sau khi triển khai phiên bản mới.",
  "Tốc độ phát hành cao đồng nghĩa với việc lỗi có thể lan truyền nhanh hơn nếu không được kiểm soát chặt chẽ.",
  "Pipeline dạng mã (pipeline as code) cho phép quy trình được tái tạo nhất quán và xem xét như mã nguồn thông thường.",
  "Blue-green deploy chạy song song hai môi trường giống hệt để chuyển mạch nhanh khi có vấn đề.",
  "Canary deploy phát hành phiên bản mới cho một nhóm nhỏ người dùng trước khi mở rộng ra toàn bộ.",
  "Rollback tự động là khả năng quay về phiên bản trước nếu chỉ số sức khỏe dịch vụ suy giảm sau khi triển khai.",
 ],
},
{
 "id":"python-gil", "name":"GIL và lập trình bất đồng bộ Python",
 "query":"GIL ảnh hưởng đa luồng thế nào?",
 "chunks":[
  "Global Interpreter Lock (GIL) là một khóa ở cấp trình thông dịch của CPython, cho phép chỉ một luồng thực thi mã bytecode Python tại một thời điểm, ngay cả trên máy có nhiều nhân xử lý. GIL tồn tại vì lý do lịch sử và để đơn giản hóa việc quản lý bộ nhớ, khi bộ đếm tham chiếu của các đối tượng cần được bảo vệ khỏi truy cập đồng thời. Hệ quả là các chương trình đa luồng xử lý nặng về tính toán không tận dụng được nhiều nhân, vì các luồng thay phiên nhau chiếm GIL thay vì chạy thực sự song song. Tuy nhiên, với các tác vụ bị chặn bởi nhập xuất như đọc tệp hay gọi mạng, luồng sẽ nhả GIL trong lúc chờ, cho phép luồng khác chạy tiếp, nên đa luồng vẫn hiệu quả cho các tác vụ chủ yếu chờ I/O.",
  "Đối với các tác vụ nặng về CPU, giải pháp thường dùng là module multiprocessing để tạo nhiều tiến trình, mỗi tiến trình có GIL riêng và có thể chạy song song thực sự trên nhiều nhân. Gần đây, lập trình bất đồng bộ với asyncio trở thành lựa chọn phổ biến cho các ứng dụng mạng hiệu năng cao, dùng vòng lặp sự kiện để xử lý nhiều tác vụ I/O trên một luồng mà không cần tạo thêm luồng hệ điều hành. asyncio phù hợp khi khối lượng công việc chủ yếu là chờ đợi, còn multiprocessing phù hợp khi cần tính toán song song. Việc chọn đúng mô hình đồng thời dựa trên đặc điểm công việc là kỹ năng quan trọng giúp tối ưu hiệu năng ứng dụng Python trong thực tế.",
 ],
 "extra":[
  "Tránh kỳ vọng sai về khả năng tăng tốc của đa luồng khi xử lý tác vụ nặng về CPU trong CPython.",
  "Thư viện mở rộng viết bằng C có thể nhả GIL khi chạy tính toán nặng, cho phép đa luồng hiệu quả hơn.",
  "Vòng lặp sự kiện của asyncio chỉ chạy trên một luồng nhưng xử lý được hàng nghìn kết nối I/O đồng thời.",
  "Multiprocessing có chi phí khởi tạo tiến trình cao hơn luồng nhưng tránh được giới hạn của GIL.",
  "Bộ đếm tham chiếu là lý do chính khiến việc bỏ GIL gặp khó khăn trong CPython cổ điển.",
  "Chọn mô hình đồng thời phù hợp đòi hỏi phân tích rõ tác vụ thiên về tính toán hay thiên về nhập xuất.",
 ],
},
{
 "id":"redis-cache", "name":"Redis và chiến lược cache",
 "query":"Chiến lược hủy cache phổ biến nào?",
 "chunks":[
  "Redis là một kho dữ liệu trong bộ nhớ, thường được dùng làm cache để giảm tải cho cơ sở dữ liệu chính và tăng tốc độ phản hồi. Vì dữ liệu được lưu trên RAM, thao tác đọc ghi trong Redis rất nhanh, thường chỉ mất dưới một phần nghìn giây. Khi dùng Redis làm cache, một câu hỏi trung tâm là làm thế nào để giữ cho dữ liệu trong cache luôn đồng bộ với nguồn dữ liệu gốc, vì cache về bản chất là một bản sao tạm thời và có thể trở nên lỗi thời. Các chiến lược làm mất hiệu lực cache giải quyết vấn đề này theo những cách khác nhau. Chiến lược write-through ghi dữ liệu vào cache và cơ sở dữ liệu cùng lúc, đảm bảo cache luôn mới nhưng tăng độ trễ ghi.",
  "Chiến lược cache-aside (lazy loading) là cách phổ biến nhất: ứng dụng đọc cache trước, nếu không có thì đọc cơ sở dữ liệu rồi ghi vào cache. Cách này đơn giản và chỉ lưu những dữ liệu thực sự được dùng, nhưng có thể gây hiện tượng cache stampede khi nhiều yêu cầu đồng loạt truy cập một khóa vừa hết hạn. Để tránh dữ liệu cũ, người ta kết hợp thời gian sống (TTL) với việc chủ động xóa hoặc cập nhật cache ngay khi dữ liệu gốc thay đổi. Việc chọn thời gian TTL hợp lý cần cân bằng giữa độ tươi của dữ liệu và tỷ lệ trúng cache, vì TTL quá dài khiến dữ liệu cũ lưu lâu còn quá ngắn khiến cache ít phát huy tác dụng.",
 ],
 "extra":[
  "Write-back ghi vào cache trước rồi mới đồng bộ cơ sở dữ liệu, cho tốc độ ghi cao nhưng có rủi ro mất dữ liệu nếu cache gặp sự cố.",
  "Cache stampede xảy ra khi nhiều yêu cầu đồng thời tìm một khóa vừa hết hạn và cùng đổ về cơ sở dữ liệu.",
  "Một hệ thống cache tốt cần giám sát tỷ lệ trúng và độ trễ để điều chỉnh chiến lược cho phù hợp từng loại dữ liệu.",
  "Dữ liệu ít thay đổi có thể dùng TTL dài, trong khi dữ liệu biến động nhanh cần TTL ngắn hoặc hủy chủ động.",
  "Chủ động xóa cache ngay khi dữ liệu gốc thay đổi giúp giảm nguy cơ trả về dữ liệu lỗi thời.",
  "Tỷ lệ trúng cache cao là chỉ số quan trọng cho thấy cache đang phát huy hiệu quả giảm tải.",
 ],
},
{
 "id":"mongodb", "name":"MongoDB và mô hình tài liệu",
 "query":"Khi nào nên dùng MongoDB?",
 "chunks":[
  "MongoDB là cơ sở dữ liệu hướng tài liệu, lưu dữ liệu dưới dạng các tài liệu JSON linh hoạt thay vì các bảng với cột cố định. Điều này cho phép lược đồ thay đổi linh hoạt theo thời gian, phù hợp khi cấu trúc dữ liệu chưa xác định rõ hoặc thường xuyên biến đổi. Trong mô hình tài liệu, dữ liệu liên quan thường được nhúng vào cùng một tài liệu để có thể đọc bằng một thao tác duy nhất, giảm số lượng truy vấn cần thực hiện. Điều này mang lại lợi thế lớn cho các ứng dụng mà dữ liệu thường được truy cập cùng nhau, chẳng hạn hồ sơ người dùng kèm danh sách địa chỉ. Tuy nhiên, việc nhúng dữ liệu cũng có giới hạn, vì tài liệu có kích thước tối đa và việc cập nhật dữ liệu lặp lại trong nhiều tài liệu có thể gây khó khăn cho tính nhất quán.",
  "MongoDB hỗ trợ chỉ mục trên bất kỳ trường nào trong tài liệu, bao gồm cả chỉ mục tổng hợp và chỉ mục toàn văn, giúp tăng tốc truy vấn đáng kể. Khả năng mở rộng theo chiều ngang thông qua sharding cho phép phân tán dữ liệu trên nhiều máy chủ, phù hợp với các hệ thống có khối lượng dữ liệu và lưu lượng lớn. Mô hình sao chép (replica set) cung cấp khả năng dự phòng và chuyển đổi dự phòng tự động khi nút chính gặp sự cố. Về tính nhất quán, MongoDB cho phép điều chỉnh mức độ giữa tốc độ và độ an toàn của thao tác ghi, từ ghi không chờ xác nhận đến ghi đa số.",
 ],
 "extra":[
  "Khi dữ liệu có mối quan hệ phức tạp và cần tính nhất quán mạnh giữa các thực thể, cơ sở dữ liệu quan hệ thường là lựa chọn an toàn hơn.",
  "Việc lựa chọn MongoDB hay cơ sở dữ liệu quan hệ nên dựa trên đặc điểm dữ liệu và mẫu truy cập cụ thể.",
  "Không nên coi một công nghệ là tốt hơn trong mọi trường hợp, mà cần đánh giá theo bài toán thực tế.",
  "Tài liệu MongoDB có giới hạn kích thước tối đa, nên dữ liệu quá lớn cần được tách hoặc lưu tham chiếu.",
  "Replica set tự động bầu nút chính mới khi nút chính hiện tại không phản hồi, tăng độ sẵn sàng.",
  "Sharding yêu cầu chọn khóa phân mảnh hợp lý để dữ liệu phân bố đều, tránh hiện tượng shard nóng.",
 ],
},
{
 "id":"microservices", "name":"Kiến trúc microservices",
 "query":"API gateway và saga là gì?",
 "chunks":[
  "Kiến trúc microservices chia một ứng dụng lớn thành nhiều dịch vụ nhỏ, độc lập, mỗi dịch vụ đảm nhiệm một chức năng nghiệp vụ cụ thể và có thể được triển khai, mở rộng riêng. Lợi ích chính là khả năng phát triển song song giữa các đội và mở rộng độc lập từng phần có tải cao, nhưng đi kèm là sự phức tạp trong giao tiếp và quản lý dữ liệu phân tán. Khi số lượng dịch vụ tăng, client không nên gọi trực tiếp từng dịch vụ vì sẽ phải xử lý nhiều địa chỉ khác nhau và nghiệp vụ phức tạp. API gateway ra đời như một điểm vào duy nhất, tiếp nhận yêu cầu từ client, định tuyến đến dịch vụ phù hợp, đồng thời đảm nhiệm các chức năng chung như xác thực, giới hạn tốc độ và tổng hợp phản hồi từ nhiều dịch vụ.",
  "Một thách thức lớn của microservices là duy trì tính nhất quán dữ liệu khi một giao dịch nghiệp vụ trải dài trên nhiều dịch vụ, vì không thể dùng giao dịch ACID xuyên suốt như trong kiến trúc nguyên khối. Saga pattern giải quyết vấn đề này bằng cách chia giao dịch lớn thành chuỗi các giao dịch cục bộ, mỗi dịch vụ thực hiện phần việc của mình và phát ra sự kiện kích hoạt bước tiếp theo. Nếu một bước thất bại, saga thực hiện các giao dịch bù trừ để hoàn tác các bước đã hoàn thành, đưa hệ thống về trạng thái nhất quán cuối cùng.",
 ],
 "extra":[
  "Có hai cách điều phối saga: choreography dùng sự kiện trao đổi trực tiếp, còn orchestration dùng bộ điều phối trung tâm.",
  "Việc chọn cách điều phối saga phụ thuộc vào độ phức tạp của luồng nghiệp vụ và nhu cầu quan sát toàn bộ quy trình.",
  "API gateway còn đảm nhiệm xác thực tập trung, giúp các dịch vụ nội bộ không phải tự xử lý bảo mật.",
  "Tính nhất quán cuối cùng chấp nhận trạng thái tạm thời không đồng bộ để đổi lấy khả năng mở rộng.",
  "Giao dịch bù trừ là thao tác ngược lại nhằm hoàn tác một giao dịch cục bộ đã hoàn thành trước đó.",
  "Microservices tăng độ phức tạp vận hành, nên cần hạ tầng giám sát và triển khai tự động phù hợp.",
 ],
},
{
 "id":"observability", "name":"Observability: log, metric, trace",
 "query":"Ba trụ cột observability là gì?",
 "chunks":[
  "Observability là khả năng hiểu được trạng thái bên trong của một hệ thống dựa trên dữ liệu mà nó phát ra từ bên ngoài, đặc biệt quan trọng khi vận hành các hệ thống phân tán phức tạp. Ba trụ cột chính của observability là log, metric và trace. Log là các bản ghi dạng văn bản mô tả sự kiện xảy ra tại một thời điểm, cung cấp chi tiết về những gì đã xảy ra nhưng khó dùng để phát hiện xu hướng. Metric là các giá trị số đo lường theo chuỗi thời gian như độ trễ, tỷ lệ lỗi hay mức sử dụng CPU, cho phép theo dõi xu hướng và thiết lập cảnh báo khi vượt ngưỡng. Trace ghi lại toàn bộ hành trình của một yêu cầu xuyên qua nhiều dịch vụ, giúp xác định chính xác dịch vụ nào gây ra độ trễ hoặc lỗi.",
  "Trong thực tế, một hệ thống quan sát tốt cần kết hợp cả ba loại dữ liệu một cách nhất quán. Khi xảy ra sự cố, metric giúp nhanh chóng khoanh vùng thời điểm và dịch vụ bất thường, trace giúp truy theo chuỗi gọi để tìm nguyên nhân gốc, còn log cung cấp chi tiết cuối cùng để xác nhận giả thuyết. Việc gắn một mã định danh theo dõi (trace id) xuyên suốt các dịch vụ là kỹ thuật quan trọng để nối các log và trace lại với nhau. Các công cụ như Prometheus cho metric, ELK cho log và Jaeger cho trace là những lựa chọn mã nguồn mở phổ biến.",
 ],
 "extra":[
  "Đầu tư vào observability từ sớm giúp giảm đáng kể thời gian khắc phục sự cố và tăng độ tin cậy của hệ thống.",
  "Metric phù hợp cho cảnh báo tự động vì có thể so sánh với ngưỡng và phát hiện bất thường theo thời gian.",
  "Trace id là khóa liên kết cho phép nối log và trace của cùng một yêu cầu xuyên suốt nhiều dịch vụ.",
  "Prometheus dùng mô hình kéo để thu thập metric, còn ELK stack gồm Elasticsearch, Logstash và Kibana cho log.",
  "Ba loại dữ liệu bổ trợ lẫn nhau và phục vụ các mục đích chẩn đoán khác nhau trong vận hành.",
  "Khoanh vùng sự cố nhanh nhờ metric, truy nguyên nhân gốc bằng trace, xác nhận bằng log là quy trình phổ biến.",
 ],
},
{
 "id":"rest-grpc", "name":"REST và gRPC",
 "query":"So sánh REST và gRPC?",
 "chunks":[
  "REST (Representational State Transfer) là phong cách kiến trúc API dùng giao thức HTTP và các phương thức chuẩn như GET, POST, PUT, DELETE để thao tác trên tài nguyên được định danh bằng URL. REST dễ hiểu, dễ tích hợp và được hỗ trợ rộng rãi bởi mọi ngôn ngữ lập trình cũng như công cụ, nên phù hợp cho các API công khai và giao tiếp giữa trình duyệt với máy chủ. Dữ liệu trao đổi thường ở dạng JSON, dễ đọc và dễ gỡ lỗi. Tuy nhiên, REST cũng có hạn chế: việc mô hình hóa một số thao tác không phải CRUD trở nên gượng ép, và mỗi yêu cầu HTTP mang theo khá nhiều chi phí chung. Khi giao tiếp giữa các dịch vụ nội bộ đòi hỏi hiệu năng cao và kiểu dữ liệu chặt chẽ, REST đôi khi không phải là lựa chọn tối ưu.",
  "gRPC là một framework gọi thủ tục từ xa do Google phát triển, dùng Protocol Buffers để định nghĩa hợp đồng dịch vụ với kiểu dữ liệu tường minh và truyền tải dưới dạng nhị phân. Nhờ định dạng nhị phân gọn nhẹ và khả năng ghép kênh trên HTTP/2, gRPC thường nhanh hơn và tốn ít băng thông hơn REST trong các luồng giao tiếp nội bộ. gRPC còn hỗ trợ streaming bốn chiều, phù hợp cho các kịch bản dữ liệu thời gian thực. Đánh đổi là gRPC khó gỡ lỗi hơn do dữ liệu nhị phân, và việc gọi trực tiếp từ trình duyệt cần lớp chuyển đổi như gRPC-Web.",
 ],
 "extra":[
  "Một chiến lược phổ biến là dùng REST cho API đối ngoại để dễ tích hợp, còn gRPC cho giao tiếp nội bộ giữa microservices.",
  "Protocol Buffers định nghĩa kiểu dữ liệu tường minh và sinh mã cho nhiều ngôn ngữ từ một tệp định nghĩa chung.",
  "HTTP/2 cho phép ghép kênh nhiều luồng trên một kết nối, khác với HTTP/1.1 xử lý tuần tự từng yêu cầu.",
  "Streaming bốn chiều của gRPC hỗ trợ cả client và server gửi nhiều thông điệp trong một phiên gọi.",
  "gRPC-Web là lớp trung gian để trình duyệt có thể gọi dịch vụ gRPC mà không cần hỗ trợ HTTP/2 đầy đủ.",
  "JSON dễ đọc và dễ gỡ lỗi hơn dữ liệu nhị phân, là lý do REST được ưa chuộng cho API công khai.",
 ],
},
{
 "id":"cdn", "name":"CDN và cache biên",
 "query":"CDN tăng tốc web thế nào?",
 "chunks":[
  "Mạng phân phối nội dung (CDN) là một hệ thống các máy chủ phân bố ở nhiều vị trí địa lý, nhằm phục vụ nội dung tĩnh và động từ máy chủ gần người dùng nhất. Khi một người dùng yêu cầu tải một trang web, thay vì truy cập trực tiếp máy chủ gốc ở xa, yêu cầu được định tuyến tới điểm biên của CDN gần vị trí người dùng, giúp giảm độ trễ mạng đáng kể. Nội dung tĩnh như ảnh, tệp CSS và JavaScript thường được cache tại các điểm biên, nên các lần truy cập sau được phục vụ ngay từ cache mà không cần về máy chủ gốc. Điều này vừa tăng tốc độ tải trang cho người dùng, vừa giảm tải cho hạ tầng máy chủ gốc. Ngoài ra, CDN còn hấp thụ các đợt tăng đột biến lưu lượng và chống lại một phần các cuộc tấn công từ chối dịch vụ phân tán.",
  "Để cache hiệu quả, CDN dựa vào các tiêu đề HTTP như Cache-Control và ETag để xác định nội dung nào được phép lưu và trong bao lâu. Tiêu đề Cache-Control với chỉ thị max-age cho biết thời gian một tài nguyên được coi là mới trước khi cần kiểm tra lại máy chủ gốc. Khi nội dung thay đổi, người quản trị có thể dùng cơ chế hủy cache (cache invalidation) để loại bỏ ngay các bản sao cũ trên các điểm biên. Một kỹ thuật phổ biến là gắn dấu vân tay vào tên tệp tĩnh, chẳng hạn thêm hash nội dung vào tên tệp, để mỗi lần nội dung đổi thì tên tệp cũng đổi và trình duyệt cũng như CDN tự động lấy bản mới.",
 ],
 "extra":[
  "Việc cấu hình đúng thời gian cache và chiến lược hủy cache là yếu tố then chốt để CDN phát huy tối đa hiệu quả.",
  "Cache-Control với max-age là chỉ thị cơ bản cho biết tài nguyên được coi là mới trong bao lâu trước khi kiểm tra lại.",
  "ETag là định danh phiên bản tài nguyên, cho phép máy chủ trả về mã 304 Not Modified khi nội dung không đổi.",
  "Cache invalidation loại bỏ ngay bản sao cũ trên điểm biên khi nội dung gốc thay đổi.",
  "Gắn hash nội dung vào tên tệp khiến tên tệp đổi theo nội dung, buộc lấy bản mới khi nội dung thay đổi.",
  "CDN gần người dùng giúp giảm số chặng mạng và thời gian khứ hồi, cải thiện trải nghiệm tải trang.",
 ],
},
{
 "id":"testing", "name":"Kiểm thử phần mềm",
 "query":"Unit test khác integration test thế nào?",
 "chunks":[
  "Unit test là loại kiểm thử tập trung vào từng đơn vị mã nhỏ nhất có thể kiểm tra độc lập, thường là một hàm hoặc một phương thức. Mục tiêu của unit test là xác minh rằng mỗi đơn vị hoạt động đúng theo thiết kế trong mọi nhánh logic, bao gồm cả các trường hợp biên và đầu vào không hợp lệ. Vì phạm vi hẹp và không phụ thuộc vào các thành phần bên ngoài, unit test chạy rất nhanh, có thể thực thi hàng nghìn ca trong vài giây. Để cô lập đơn vị đang kiểm tra, người ta thường dùng kỹ thuật mock để thay thế các phụ thuộc như cơ sở dữ liệu hay dịch vụ mạng bằng các đối tượng giả có hành vi được kiểm soát. Nhờ chạy nhanh và ổn định, unit test thường là lớp bảo vệ đầu tiên và được chạy liên tục trong quy trình tích hợp liên tục.",
  "Integration test kiểm tra sự phối hợp giữa nhiều đơn vị hoặc giữa ứng dụng với các thành phần bên ngoài như cơ sở dữ liệu, hàng đợi tin nhắn hay API. Mục tiêu của integration test là phát hiện các lỗi phát sinh từ sự tương tác giữa các thành phần, chẳng hạn như sai lệch về định dạng dữ liệu hay thứ tự gọi, những vấn đề mà unit test không thể phát hiện vì chúng đã được cô lập. Integration test thường chậm hơn và phức tạp hơn do cần môi trường gần với thực tế, có thể dùng container để dựng các dịch vụ phụ thuộc. Trong thực tế, một chiến lược kiểm thử cân bằng thường theo mô hình kim tự tháp: nhiều unit test nhanh ở đáy, ít integration test hơn ở giữa, và rất ít kiểm thử đầu cuối chậm ở đỉnh.",
 ],
 "extra":[
  "Cách phân bổ kiểm thử theo mô hình kim tự tháp giúp đạt độ tin cậy cao mà vẫn giữ thời gian chạy ở mức chấp nhận được.",
  "Mock cho phép kiểm soát hành vi của phụ thuộc, giúp unit test xác định đúng đơn vị gây lỗi.",
  "Integration test phát hiện lỗi từ tương tác giữa các thành phần mà unit test không thể thấy do đã cô lập.",
  "Dùng container để dựng dịch vụ phụ thuộc giúp integration test chạy gần với môi trường thực tế.",
  "Kiểm thử đầu cuối chậm và tốn kém nên thường chỉ giới hạn ở vài luồng nghiệp vụ quan trọng.",
  "Unit test chạy liên tục trong CI là lớp bảo vệ đầu tiên chống lại hồi quy mã nguồn.",
 ],
},
{
 "id":"sharding", "name":"Sharding và nhân bản dữ liệu",
 "query":"Sharding khác replication thế nào?",
 "chunks":[
  "Sharding là kỹ thuật chia một tập dữ liệu lớn thành nhiều phần nhỏ hơn gọi là shard, mỗi shard được lưu trên một máy chủ riêng biệt. Mục tiêu của sharding là mở rộng theo chiều ngang, cho phép cơ sở dữ liệu xử lý khối lượng dữ liệu và lưu lượng vượt quá khả năng của một máy chủ đơn lẻ. Việc chọn khóa phân mảnh (shard key) là quyết định quan trọng nhất, vì nó quyết định dữ liệu được phân bố đều hay lệch giữa các shard. Một khóa phân mảnh tốt giúp dữ liệu phân tán đồng đều, tránh hiện tượng shard nóng khi quá nhiều truy vấn dồn vào một shard. Tuy nhiên, sharding làm tăng đáng kể độ phức tạp vận hành, vì các truy vấn trải dài nhiều shard khó thực hiện và cần cơ chế định tuyến cẩn thận.",
  "Replication là kỹ thuật tạo các bản sao của cùng một tập dữ liệu trên nhiều máy chủ, nhằm tăng độ sẵn sàng và khả năng chịu lỗi. Trong mô hình chủ - bản sao (primary-replica), máy chủ chính tiếp nhận thao tác ghi và nhân bản sang các bản sao chỉ phục vụ đọc. Nhờ đó, hệ thống có thể phân tán tải đọc và vẫn hoạt động khi máy chủ chính gặp sự cố bằng cách thăng cấp một bản sao lên làm chính. Sự khác biệt cốt lõi giữa sharding và replication nằm ở mục đích: sharding chia dữ liệu để tăng dung lượng và khả năng ghi, còn replication nhân bản dữ liệu để tăng độ tin cậy và khả năng đọc.",
 ],
 "extra":[
  "Các hệ thống lớn thường kết hợp cả sharding và replication, vừa phân tán dữ liệu vừa nhân bản từng shard để tăng độ sẵn sàng.",
  "Sharding thường chỉ được áp dụng khi dữ liệu đã thực sự vượt quá ngưỡng mà một máy chủ có thể xử lý.",
  "Khóa phân mảnh phải phân bố đều để tránh hiện tượng shard nóng làm mất hiệu quả mở rộng.",
  "Trong mô hình chủ - bản sao, thao tác ghi tập trung vào nút chính còn các bản sao chỉ phục vụ đọc.",
  "Thăng cấp bản sao lên làm chính là cơ chế chuyển đổi dự phòng khi nút chính gặp sự cố.",
  "Kiến trúc phân tán bền vững cần vừa chịu được sự cố vừa xử lý được lưu lượng cao.",
 ],
},
]

# Câu bổ sung thêm (để chunk đạt ~512 tokens), 6 câu/topic.
EXTRA2 = {
"sql-index": [
 "So sánh chi phí ghi và lợi ích đọc giúp quyết định có nên thêm chỉ mục hay không.",
 "Theo dõi hiệu năng truy vấn định kỳ giúp phát hiện chỉ mục cần điều chỉnh hoặc loại bỏ.",
 "EXPLAIN ANALYZE cho biết chi phí thực tế từng bước, giúp xác nhận truy vấn có dùng index hay không.",
 "Chỉ mục giúp giảm độ phức tạp truy vấn từ quét toàn bộ xuống tìm kiếm có cấu trúc.",
 "Cột có độ chọn lọc cao nên được đặt ở đầu chỉ mục tổng hợp để phát huy hiệu quả.",
 "Bảo trì chỉ mục định kỳ giúp giữ hiệu năng ổn định khi dữ liệu tăng trưởng theo thời gian.",
],
"sql-vacuum": [
 "VACUUM dọn dead tuple còn ANALYZE cập nhật thống kê, hai việc bổ trợ lẫn nhau.",
 "Kết hợp VACUUM ANALYZE trong một lệnh vừa dọn dẹp vừa làm mới thống kê.",
 "Bảng phình to làm tăng thời gian quét và chiếm nhiều dung lượng đĩa hơn mức cần thiết.",
 "Theo dõi autovacuum giúp phát hiện sớm bảng bị tụt hậu về dọn dẹp dead tuple.",
 "MVCC sinh ra dead tuple nên VACUUM là công việc bảo trì thiết yếu của PostgreSQL.",
 "Thống kê chính xác giúp planner chọn được kế hoạch thực thi tối ưu.",
],
"raft-consensus": [
 "Đa số phiếu là điều kiện đủ để một nút trở thành leader trong thuật toán Raft.",
 "Nhật ký nhất quán là nền tảng cho máy trạng thái sao chép hoạt động chính xác.",
 "Raft tách bạch bầu cử, nhân bản và an toàn thành các phần độc lập dễ kiểm chứng.",
 "Hệ thống phân tán dùng Raft có thể tiếp tục hoạt động khi thiểu số nút gặp sự cố.",
 "Raft đơn giản hơn Paxos nhờ tách bạch các vấn đề con thành các cơ chế rõ ràng.",
 "Leader gửi nhịp tim định kỳ để duy trì quyền lãnh đạo và ngăn bầu cử không cần thiết.",
],
"docker": [
 "Image bất biến giúp tái tạo môi trường chạy giống hệt trên mọi máy.",
 "Layer cache giúp xây dựng lại image nhanh hơn khi chỉ thay đổi một phần nhỏ.",
 "Container cô lập giúp chạy nhiều ứng dụng trên cùng máy chủ một cách an toàn.",
 "Volume là cơ chế lưu dữ liệu bền vững bên ngoài vòng đời của container.",
 "Dockerfile khai báo các bước xây dựng image một cách tường minh và có thể tái tạo.",
 "Gộp nhiều lệnh RUN vào một giúp giảm số lớp và kích thước image cuối cùng.",
],
"kubernetes": [
 "Kubernetes tự phục hồi bằng cách tạo lại Pod khi phát hiện lỗi hoặc mất Pod.",
 "Label selector là cách Service chọn tập Pod cần cân bằng tải một cách linh hoạt.",
 "Rolling update thay thế dần từng Pod để dịch vụ không bị gián đoạn khi triển khai.",
 "Cấu hình khai báo giúp Kubernetes giữ trạng thái mong muốn mà không cần lệnh thủ công.",
 "Service cung cấp địa chỉ ổn định cho tập Pod thay đổi liên tục theo thời gian.",
 "Khai báo trạng thái mong muốn để Kubernetes tự động điều chỉnh khi có sai lệch.",
],
"tcp-http": [
 "Bắt tay ba bước là chi phí cố định mỗi khi thiết lập một kết nối TCP mới.",
 "Keep-alive giảm số lần bắt tay khi tải nhiều tài nguyên từ cùng một máy chủ.",
 "HTTP/2 ghép kênh nhiều luồng trên một kết nối, tránh hiện tượng chặn đầu dòng.",
 "Cân bằng thời gian giữ kết nối giúp tối ưu cả độ trễ lẫn tài nguyên máy chủ.",
 "Độ trễ khứ hồi là chi phí tối thiểu cho mỗi lần bắt tay thiết lập kết nối.",
 "Ghép kênh HTTP/2 cho phép nhiều yêu cầu đi song song trên một kết nối duy nhất.",
],
"tls": [
 "Khóa phiên đối xứng dùng để mã hóa dữ liệu vì nhanh hơn nhiều so với mã hóa bất đối xứng.",
 "TLS 1.3 rút gọn bắt tay còn một khứ hồi, giảm độ trễ thiết lập kết nối.",
 "Chuỗi tin cậy gồm chứng chỉ máy chủ, chứng chỉ trung gian và chứng chỉ gốc.",
 "Perfect forward secrecy bảo vệ phiên cũ ngay cả khi khóa máy chủ bị lộ.",
 "Xác minh tên miền trong chứng chỉ ngăn kẻ tấn công giả mạo danh tính máy chủ.",
 "Khóa phiên được thay mới mỗi lần kết nối để tăng cường bảo mật cho phiên làm việc.",
],
"ml-overfit": [
 "Đường cong học tập tách xa giữa train và validation là tín hiệu rõ ràng của overfitting.",
 "Chính quy hóa phạt trọng số lớn để mô hình đơn giản và tổng quát hơn.",
 "Dropout buộc mạng học biểu diễn bền vững thay vì phụ thuộc vài nơ-ron cụ thể.",
 "Tăng cường dữ liệu làm giàu tập huấn luyện, đặc biệt hiệu quả trong thị giác máy tính.",
 "Dừng sớm ngừng huấn luyện khi lỗi trên tập xác thực bắt đầu tăng trở lại.",
 "Đơn giản hóa mô hình là cách hiệu quả khi lượng dữ liệu huấn luyện hạn chế.",
],
"jwt": [
 "Payload JWT chỉ được mã hóa Base64, không phải mã hóa bí mật thực sự.",
 "Cookie HttpOnly ngăn JavaScript đọc token, giảm rủi ro từ tấn công XSS.",
 "Refresh token có thời gian sống dài hơn và lưu ở nơi an toàn hơn access token.",
 "Thu hồi token là hạn chế của JWT vì bản chất không trạng thái của nó.",
 "Access token ngắn hạn giúp giảm rủi ro khi token bị đánh cắp.",
 "HTTPS bảo vệ token khỏi bị chặn bắt trong quá trình truyền qua mạng.",
],
"owasp": [
 "Tham số hóa truy vấn tách dữ liệu khỏi mã SQL, ngăn chặn SQL injection.",
 "Mã hóa đầu ra đúng ngữ cảnh là biện pháp cốt lõi để chống XSS.",
 "CSP hạn chế nguồn script mà trình duyệt được phép thực thi.",
 "Đặc quyền tối thiểu giới hạn thiệt hại khi lỗ hổng bị khai thác.",
 "Framework hiện đại có cơ chế thoát tự động giúp giảm rủi ro XSS.",
 "Rà soát mã định kỳ giúp phát hiện sớm các lỗ hổng bảo mật còn sót lại.",
],
"git-ci": [
 "CI tự động chạy kiểm thử mỗi khi mã được đẩy lên, phát hiện lỗi sớm.",
 "Pipeline as code giúp quy trình được tái tạo và xem xét như mã nguồn thông thường.",
 "Blue-green deploy chạy song song hai môi trường để chuyển mạch nhanh khi có vấn đề.",
 "Rollback tự động quay về phiên bản trước khi chỉ số sức khỏe dịch vụ suy giảm.",
 "GitHub Actions và Jenkins cho phép định nghĩa pipeline dưới dạng mã.",
 "Kiểm thử tự động đáng tin cậy là nền tảng của tích hợp liên tục.",
],
"python-gil": [
 "Đa luồng hiệu quả cho I/O vì luồng nhả GIL trong lúc chờ đợi.",
 "Multiprocessing chạy song song thực sự trên nhiều nhân nhờ tiến trình riêng.",
 "asyncio dùng vòng lặp sự kiện xử lý nhiều tác vụ I/O trên một luồng.",
 "Chọn mô hình đồng thời cần phân tích tác vụ thiên về tính toán hay nhập xuất.",
 "GIL giới hạn khả năng song song tính toán CPU trong CPython.",
 "Thư viện viết bằng C có thể nhả GIL khi chạy tính toán nặng, giúp đa luồng hiệu quả hơn.",
],
"redis-cache": [
 "Write-through ghi đồng thời vào cache và cơ sở dữ liệu, giữ cache luôn mới.",
 "Write-back ghi cache trước rồi đồng bộ sau, nhanh nhưng có rủi ro mất dữ liệu.",
 "TTL cần cân bằng giữa độ tươi của dữ liệu và tỷ lệ trúng cache.",
 "Giám sát tỷ lệ trúng giúp đánh giá cache đang phát huy hiệu quả hay không.",
 "Cache stampede xảy ra khi nhiều yêu cầu đồng loạt truy cập khóa vừa hết hạn.",
 "Chủ động hủy cache ngay khi dữ liệu gốc thay đổi giúp giảm dữ liệu lỗi thời.",
],
"mongodb": [
 "Lược đồ linh hoạt phù hợp với dữ liệu có cấu trúc thay đổi thường xuyên.",
 "Nhúng dữ liệu liên quan giúp đọc bằng một thao tác, giảm số truy vấn cần thực hiện.",
 "Replica set dự phòng và tự chuyển đổi khi nút chính gặp sự cố.",
 "Chọn công nghệ nên dựa trên mẫu truy cập và yêu cầu nhất quán cụ thể.",
 "Sharding mở rộng theo chiều ngang cho dữ liệu và lưu lượng lớn.",
 "Điều chỉnh mức ghi cân bằng giữa tốc độ và độ an toàn của thao tác.",
],
"microservices": [
 "API gateway là điểm vào duy nhất, định tuyến và xác thực tập trung.",
 "Saga chia giao dịch lớn thành chuỗi giao dịch cục bộ độc lập.",
 "Giao dịch bù trừ hoàn tác các bước đã hoàn thành khi saga thất bại.",
 "Tính nhất quán cuối cùng đổi lấy khả năng mở rộng và độc lập giữa các dịch vụ.",
 "Choreography trao đổi sự kiện trực tiếp mà không cần bộ điều phối trung tâm.",
 "Orchestration dùng bộ điều phối trung tâm ra lệnh cho từng dịch vụ.",
],
"observability": [
 "Metric phù hợp cho cảnh báo vì so được với ngưỡng theo thời gian.",
 "Trace id liên kết log và trace của cùng một yêu cầu xuyên suốt dịch vụ.",
 "Prometheus thu thập metric, ELK xử lý log, Jaeger theo dõi trace.",
 "Khoanh vùng bằng metric, truy nguyên bằng trace, xác nhận bằng log.",
 "Trace ghi lại toàn bộ hành trình của yêu cầu xuyên qua nhiều dịch vụ.",
 "Log cung cấp chi tiết sự kiện xảy ra tại một thời điểm cụ thể.",
],
"rest-grpc": [
 "REST dễ đọc, dễ gỡ lỗi, phù hợp cho API công khai.",
 "gRPC dùng nhị phân và HTTP/2, nhanh hơn cho giao tiếp nội bộ.",
 "Protocol Buffers định nghĩa kiểu dữ liệu và sinh mã cho nhiều ngôn ngữ.",
 "gRPC-Web là cầu nối cho trình duyệt gọi dịch vụ gRPC.",
 "gRPC khó gỡ lỗi hơn do dữ liệu được truyền dưới dạng nhị phân.",
 "REST phù hợp cho giao tiếp giữa trình duyệt với máy chủ.",
],
"cdn": [
 "CDN phục vụ nội dung từ điểm biên gần người dùng, giảm độ trễ mạng.",
 "Cache-Control và ETag quyết định nội dung được lưu trong bao lâu.",
 "Cache invalidation xóa bản sao cũ trên điểm biên khi nội dung gốc thay đổi.",
 "Gắn hash vào tên tệp buộc lấy bản mới mỗi khi nội dung thay đổi.",
 "CDN hấp thụ lưu lượng đột biến và chống lại một phần tấn công DDoS.",
 "Điểm biên gần giúp giảm số chặng mạng và thời gian khứ hồi.",
],
"testing": [
 "Mô hình kim tự tháp phân bổ unit, integration và e2e test một cách hợp lý.",
 "Mock cô lập phụ thuộc giúp unit test xác định đúng đơn vị gây lỗi.",
 "Integration test phát hiện lỗi phát sinh từ tương tác giữa các thành phần.",
 "Unit test nhanh chạy liên tục trong CI, là lớp bảo vệ đầu tiên.",
 "Integration test cần môi trường gần với thực tế, có thể dùng container.",
 "E2E test chậm nên thường chỉ giới hạn ở vài luồng nghiệp vụ quan trọng.",
],
"sharding": [
 "Sharding chia dữ liệu để tăng dung lượng và khả năng ghi.",
 "Replication nhân bản dữ liệu để tăng độ tin cậy và khả năng đọc.",
 "Kết hợp cả hai cho kiến trúc phân tán chịu lỗi và mở rộng tốt.",
 "Khóa phân mảnh phải phân bố đều để tránh hiện tượng shard nóng.",
 "Thăng cấp bản sao lên làm chính khi nút chính gặp sự cố.",
 "Kiến trúc kết hợp sharding và replication cho độ bền và khả năng mở rộng.",
],
}

# ---------- sinh dữ liệu ----------
queries = []
chunks = []
score_seed = 0

for t in TOPICS:
    q = {
        "id": f"q-{t['id']}",
        "text": t["query"],
        "topic": t["id"],
        "token_count": nt(t["query"]),
        "relevant_chunks": [],
    }
    extra = t["extra"] + EXTRA2[t["id"]]
    # phân bổ greedy: mỗi câu extra nối vào chunk đang ngắn hơn (theo token)
    texts = [t["chunks"][0], t["chunks"][1]]
    for s in extra:
        i = 0 if nt(texts[0]) <= nt(texts[1]) else 1
        texts[i] = texts[i] + " " + s
    for i, ctext in enumerate(texts):
        cid = f"c-{t['id']}-{i+1}"
        base = 0.95 - i * 0.08
        score = round(max(0.05, min(0.99, base + ((score_seed * 37) % 13) / 100.0)), 3)
        score_seed += 1
        chunks.append({
            "id": cid,
            "topic": t["id"],
            "text": ctext,
            "token_count": nt(ctext),
            "score": score,
        })
        q["relevant_chunks"].append(cid)
    queries.append(q)

data = {
    "meta": {
        "tokenizer": "cl100k_base (tiktoken)",
        "note": "Chunk ~512 tokens (tiktoken thật). Ground truth: query.relevant_chunks.",
        "n_queries": len(queries),
        "n_chunks": len(chunks),
    },
    "queries": queries,
    "chunks": chunks,
}

os.makedirs("data", exist_ok=True)
out = "data/sample_inputs.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

qt = [q["token_count"] for q in queries]
ct = [c["token_count"] for c in chunks]
print(f"Wrote {out}")
print(f"queries: {len(qt)}  (min={min(qt)} max={max(qt)} avg={sum(qt)/len(qt):.1f})")
print(f"chunks : {len(ct)}  (min={min(ct)} max={max(ct)} avg={sum(ct)/len(ct):.1f})")
q_over = [q["id"] for q in queries if q["token_count"] > 20]
print(f"queries >20 tokens: {q_over}")
print(f"chunks <480: {[c['id'] for c in chunks if c['token_count']<480]}")
print(f"chunks >560: {[c['id'] for c in chunks if c['token_count']>560]}")
