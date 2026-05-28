package com.pq.agent_service.domain.dto;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

@Data
@ApiModel(description = "消息dto实体")
public class MessageDTO {
    @ApiModelProperty("会话id")
    private String sessionId;
    @ApiModelProperty("消息内容")
    private String content;
    @ApiModelProperty("消息类型")
    //1:文本  2：图片
    private Integer messageType;

}