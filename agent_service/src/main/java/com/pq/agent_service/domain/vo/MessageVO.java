package com.pq.agent_service.domain.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@ApiModel(description = "会话VO实体")
public class MessageVO {
    @ApiModelProperty("消息id ")
    private Long id;

    @ApiModelProperty("会话id ")
    private String sessionId;

    @ApiModelProperty("消息内容")
    private String content;

    @ApiModelProperty("发送者类型 ")
    private Integer senderType;

    @ApiModelProperty("消息类型 ")
    private Integer messageType;

    @ApiModelProperty("创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "Asia/Shanghai")
    private LocalDateTime createTime;
}
