package com.pq.agent_service.domain.po;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.time.LocalDateTime;
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("service_message")
public class Message {
    @TableId(type = IdType.AUTO)
    private Long id;

    // TEXT 类型直接用 String
    @TableField("content")
    private String content;

    @TableField("session_id")
    private String sessionId;

    @TableField("sender_type")
    private Integer senderType; //1:用户  2：客服(ai)

    @TableField("message_type")
    private Integer messageType ; //1:文本  2：图片

    @TableField("create_time")
    private LocalDateTime createTime;

    // ... 其他字段
}
